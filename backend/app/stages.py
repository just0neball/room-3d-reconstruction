"""Maps the fine-grained stage names emitted by Person A's scripts onto the four
user-facing stages the progress bar shows (extracting frames, estimating poses,
training, exporting)."""

# Names come from send_progress(...) calls in pipeline/*.py
STAGE_GROUPS = {
    "queued": "queued",
    "starting": "starting",
    "extracting_frames": "extract",
    "filtering_frames": "extract",
    "estimating_poses": "poses",
    "estimating_poses_vggt": "poses",
    "preparing_training_data": "poses",  # COLMAP undistortion + dataset prep
    "training_gaussians": "train",
    "exporting": "export",
    "packaging": "export",
    "uploading_results": "export",
    "done": "done",
    "downloading_results": "export",
    # Week 2
    "loading_segmenter": "track", "tracking_frames": "track",
    "loading_diffusion": "edit", "editing_reference": "edit", "editing_frames": "edit",
    "preparing_dataset": "rebuild", "rebuilding_splat": "rebuild",
}

GROUP_LABELS = {
    "queued": "Queued",
    "starting": "Starting worker",
    "extract": "Extracting frames",
    "poses": "Estimating camera poses",
    "train": "Training Gaussian splat",
    "export": "Exporting scene",
    "done": "Done",
    "track": "Tracking the object",
    "edit": "Diffusion editing",
    "rebuild": "Rebuilding the splat",
}

# Order for a stepper UI
PIPELINE_GROUPS = ["extract", "poses", "train", "export"]


def group_for(stage: str) -> str:
    return STAGE_GROUPS.get(stage, stage)


def label_for(stage: str) -> str:
    if stage in LABELS_BY_STAGE:
        return LABELS_BY_STAGE[stage]
    group = group_for(stage)
    return GROUP_LABELS.get(group, stage.replace("_", " ").capitalize())


LABELS_BY_STAGE = {
    "loading_segmenter": "Loading segmentation model", "tracking_frames": "Finding the object in every frame",
    "loading_diffusion": "Loading diffusion model", "editing_reference": "Editing the reference frame",
    "editing_frames": "Editing all frames", "preparing_dataset": "Preparing edited dataset",
    "rebuilding_splat": "Rebuilding the 3D scene", "downloading_results": "Downloading results",
}
