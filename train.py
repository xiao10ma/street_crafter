import os
import torch
import random
import time
from random import randint
from create_scene import create_scene
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.models.scene import Scene
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.utils.loss_utils import l1_loss, l2_loss, psnr, ssim, huber
from street_gaussian.utils.img_utils import save_img_torch, visualize_depth_numpy
from street_gaussian.utils.general_utils import safe_state
from street_gaussian.utils.camera_utils import Camera
from street_gaussian.utils.cfg_utils import save_cfg
from street_gaussian.config import cfg
from street_gaussian.utils.lpipsPyTorch import lpips
from tqdm import tqdm
from street_gaussian.utils.system_utils import searchForMaxIteration
from street_gaussian.utils.diffusion_utils import getDiffusionRunner
from argparse import ArgumentParser, Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import shutil
import pathlib

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_stop, profiler_step

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore', "submodules", "video_diffusion", "nvs_solver"]
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)
    log_dir = pathlib.Path(__file__).parent.resolve()
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data
    diffusion_args = cfg.diffusion

    tb_writer = prepare_output_and_logger()

    scene: Scene = create_scene()
    gaussians: StreetGaussianModel = scene.gaussians
    gaussians.training_setup()
    gaussians_renderer = StreetGaussianRenderer()

    use_diffusion = diffusion_args.use_diffusion and len(diffusion_args.sample_iterations) > 0
    if use_diffusion:
        diffusion_runner = getDiffusionRunner(scene)
        print("Diffusion model loaded")

    # Load checkpoint if it exists
    if len(os.listdir(cfg.trained_model_dir)) > 0:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    else:
        start_iter = 1
        print('No checkpoint found. Training from scratch')

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    progress_bar = tqdm(range(start_iter, training_args.iterations + 1), desc='Starting training...')

    viewpoint_stack = []
    train_viewpoint_stack: List[Camera] = scene.getTrainCameras().copy()
    viewpoint_stack += train_viewpoint_stack
    # Move training images to GPU
    for viewpoint in train_viewpoint_stack:
        viewpoint.set_device('cuda')

    training_camera_number = len(viewpoint_stack)

    min_scale = min(diffusion_args.sample_scales)
    max_scale = max(diffusion_args.sample_scales)
    min_iteration = min(diffusion_args.sample_iterations)
    max_iteration = max(diffusion_args.sample_iterations)

    # Perform the actual training procedure
    for iteration in range(start_iter, training_args.iterations + 1):
        profiler_step()

        iter_start.record()  # type: ignore
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        restarting = iteration == start_iter and iteration - 1 in diffusion_args.sample_iterations
        if use_diffusion and (iteration in diffusion_args.sample_iterations or restarting):
            with torch.no_grad():
                print(f"Sampling diffusion at iteration {iteration - restarting}")
                novel_viewpoint_stack = scene.getNovelViewCameras().copy()

                scale = (min_scale - max_scale) * (iteration - restarting - min_iteration) / (max_iteration - min_iteration) + max_scale

                diffusion_result = diffusion_runner.run(novel_viewpoint_stack, train_viewpoint_stack, use_render=True, scale=scale, masked_guidance=iteration >= cfg.diffusion.masked_guidance_iter)  # type: ignore

                # Move training images to GPU
                viewpoint_stack: List[Camera] = []  # remove the previous sampling results
                viewpoint_stack += train_viewpoint_stack
                for viewpoint in novel_viewpoint_stack:
                    if not viewpoint.meta['skip_camera']:
                        viewpoint.set_device('cuda')
                        viewpoint_stack.append(viewpoint)

                # novel_camera_number = len(viewpoint_stack) - training_camera_number

        should_sample_novel_view = random.choices([0, 1], [1 - cfg.train.novel_view_prob, cfg.train.novel_view_prob])[0]

        if should_sample_novel_view and len(viewpoint_stack) > training_camera_number:
            viewpoint_cam: Camera = viewpoint_stack[randint(training_camera_number, len(viewpoint_stack) - 1)]
        else:
            viewpoint_cam: Camera = viewpoint_stack[randint(0, training_camera_number - 1)]

        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.guidance['mask'] if 'mask' in viewpoint_cam.guidance else torch.ones_like(gt_image[0:1]).bool()
        sky_mask = viewpoint_cam.guidance['sky_mask'] if 'sky_mask' in viewpoint_cam.guidance else None
        obj_bound = viewpoint_cam.guidance['obj_bound'] if 'obj_bound' in viewpoint_cam.guidance else None

        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth']  # [1, H, W]
        scalar_dict = dict()
        if viewpoint_cam.meta['is_novel_view']:
            assert viewpoint_cam.meta['diffusion_original_image'] is not None, 'Diffusion original image is not found'
            image = diffusion_runner.preprocess_tensor(image)  # type: ignore
            mask = diffusion_runner.preprocess_tensor(mask)  # type: ignore
            gt_image = viewpoint_cam.meta['diffusion_original_image']

            upper = int(mask.shape[-2] * 0.4)
            mask[..., :upper, :] = 0 # do not compute loss on the upper half of the image?
            gt_image = viewpoint_cam.meta['diffusion_original_image']            
            image_loss = image[:, upper:, :]
            gt_image_loss = gt_image[:, upper:, :]
            mask_loss = mask[:, upper:, :]            
            Ll1 = l1_loss(image_loss, gt_image_loss, mask_loss)
            ssim_value = ssim(image_loss, gt_image_loss, mask=mask_loss)
            lpips_value = lpips(image_loss * mask_loss, gt_image_loss * mask_loss)
            # Ll1 = l1_loss(image, gt_image, mask)
            # ssim_value = ssim(image, gt_image, mask=mask)
            # lpips_value = lpips(image * mask, gt_image * mask)
            loss = (1.0 - optim_args.lambda_novel_dssim) * optim_args.lambda_novel_l1 * Ll1 + optim_args.lambda_novel_dssim * (1.0 - ssim_value) + optim_args.lambda_novel_lpips * lpips_value
            loss = loss * optim_args.lambda_novel
            scalar_dict['novel_lpips'] = lpips_value.item()
            scalar_dict['novel_ssim'] = ssim_value.item()
            scalar_dict['novel_l1'] = Ll1.item()
        else:
            # rgb loss
            Ll1 = l1_loss(image, gt_image, mask)
            ssim_value = ssim(image, gt_image, mask=mask)
            lpips_value = lpips(image * mask, gt_image * mask)
            loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim_value) + optim_args.lambda_lpips * lpips_value
            scalar_dict['l1_loss'] = Ll1.item()
            scalar_dict['ssim'] = ssim_value.item()
            scalar_dict['lpips'] = lpips_value.item()

            # These regualrizations will not take effect since gts are not present in the guidance dict
            # sky loss
            if optim_args.lambda_sky > 0 and (gaussians.include_sky or gaussians.include_cube_map) and sky_mask is not None:
                acc = torch.clamp(acc, min=1e-6, max=1. - 1e-6)
                # sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
                sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -(acc * torch.log(acc) + (1. - acc) * torch.log(1. - acc))).mean()
                if len(optim_args.lambda_sky_scale) > 0:
                    sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
                scalar_dict['sky_loss'] = sky_loss.item()
                loss += optim_args.lambda_sky * sky_loss

            if optim_args.lambda_reg > 0 and gaussians.include_obj and obj_bound is not None and iteration % cfg.train.reg_obj_acc_every and iteration > cfg.optim.densify_until_iter:
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
                acc_obj = torch.clamp(acc_obj, min=1e-6, max=1. - 1e-6)
                obj_acc_loss = torch.where(obj_bound, -(acc_obj * torch.log(acc_obj) + (1. - acc_obj) * torch.log(1. - acc_obj)), -torch.log(1. - acc_obj)).mean()
                scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
                loss += optim_args.lambda_reg * obj_acc_loss

            # lidar depth loss
            if optim_args.lambda_depth_lidar > 0 and 'lidar_depth' in viewpoint_cam.guidance:
                lidar_depth = viewpoint_cam.guidance['lidar_depth']
                depth_mask = torch.logical_and((lidar_depth > 0.), mask)
                # expected_depth = depth / (render_pkg['acc'] + 1e-10)
                depth_error = torch.abs((depth[depth_mask] - lidar_depth[depth_mask]))

                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
                loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                
            # scaling loss
            if optim_args.lambda_scale_flatten > 0:
                scaling = gaussians.get_scaling 
                scaling_reg_loss = scaling.min(dim=1)[0].mean() + ((scaling.topk(2)[0]**2).sum(1)/scaling.topk(2)[0].prod(1) -2).mean()
                scalar_dict['scaling_reg_loss'] = scaling_reg_loss.item()
                loss += optim_args.lambda_scale_flatten * scaling_reg_loss
                
            # color correction loss
            if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
                color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)  # type: ignore
                scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
                loss += optim_args.lambda_color_correction * color_correction_reg_loss

        scalar_dict['loss'] = loss.item()
        loss.backward()

        iter_end.record()  # type: ignore

        with torch.no_grad():

            is_save_images = True
            if is_save_images and (iteration % 1000 == 0):
                # row0: gt_image, image, depth
                # row1: acc, image_obj, acc_obj
                depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
                depth_colored = depth_colored[..., [2, 1, 0]] / 255.
                depth_colored = torch.from_numpy(depth_colored).to(device='cuda', dtype=torch.float, non_blocking=True).permute(2, 0, 1)

                acc = acc.repeat(3, 1, 1)
                with torch.no_grad():
                    render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                    image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
                acc_obj = acc_obj.repeat(3, 1, 1)

                if viewpoint_cam.meta['is_novel_view']:
                    depth_colored = diffusion_runner.preprocess_tensor(depth_colored)  # type: ignore
                    acc = diffusion_runner.preprocess_tensor(acc)  # type: ignore
                    image_obj = diffusion_runner.preprocess_tensor(image_obj)  # type: ignore
                    acc_obj = diffusion_runner.preprocess_tensor(acc_obj)  # type: ignore

                row0 = torch.cat([gt_image, image, depth_colored], dim=2)
                row1 = torch.cat([acc, image_obj, acc_obj], dim=2)
                image_to_show = torch.cat([row0, row1], dim=1)
                image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
                os.makedirs(f"{cfg.model_path}/log_images", exist_ok=True)
                save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")

            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:
                # Progress bar
                if not viewpoint_cam.meta['is_novel_view']:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                progress_bar.set_description(f"Exp: {cfg.task}-{cfg.exp_name}, Loss: {ema_loss_for_log:.{7}f}, PSNR: {ema_psnr_for_log:.{4}f}")

            progress_bar.update(1)

            # Densification
            if iteration < optim_args.densify_until_iter:
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, viewpoint_cam)
                if 'viewspace_points_sky' in render_pkg:
                    viewspace_point_tensor_sky = render_pkg['viewspace_points_sky']
                    visibility_filter_sky = render_pkg['visibility_filter_sky'] 
                    radii_sky = render_pkg['radii_sky']
                    gaussians.set_max_radii2D_sky(radii_sky, visibility_filter_sky)
                    gaussians.add_densification_stats_sky(viewspace_point_tensor_sky, visibility_filter_sky, viewpoint_cam)
            
                prune_big_points = iteration > optim_args.opacity_reset_interval and optim_args.prune_big_points

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)  # type: ignore
                        tensor_dict.update(tensors)  # type: ignore

            # Reset opacity
            if iteration < optim_args.densify_until_iter and iteration > optim_args.densify_from_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            try:
                training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)
            except Exception as e:
                print(f'Failed to perform training report: {red(e)}')

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                try:
                    state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                    state_dict['iter'] = iteration + 1
                    ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                    torch.save(state_dict, ckpt_path)
                except Exception as e:
                    print(f'Failed to save checkpoint: {red(e)}')
                    if iteration == training_args.checkpoint_iterations[-1]:
                        stacktrace()
                        stop_prog()  # stop it, otherwise multiple lives


def prepare_output_and_logger():

    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path'] = cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except Exception as e:
            print(f'Failed to write to tensorboard: {red(e)}')

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras': scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.to('cuda', non_blocking=True).bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Save runtime code
    try:
        saveRuntimeCode(cfg.model_path + '/runtime_code')
    except:
        print('Failed to save runtime code')

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    setup_profiler(
        cfg.profiler.enabled,
        cfg.profiler.skip_first,
        cfg.profiler.wait,
        cfg.profiler.warmup,
        cfg.profiler.active,
        cfg.profiler.repeat,
        record_dir=cfg.record_dir
    )
    profiler_start()
    catch_throw(training)()
    profiler_stop()
    # All done
    print("\nTraining complete.")
