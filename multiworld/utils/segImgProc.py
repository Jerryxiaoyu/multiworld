from multiworld.utils.imgproc import get_bbox_cropped_image_path, mask2bbox, get_bbox_cropped_image_raw
import numpy as np

def convert2mask(segmask, body_id, num_object=3):
    mask_bias = 3
    mask_body_index_list = [_ + mask_bias for _ in range(num_object)]
    body_mask_index = body_id

    mask = segmask.copy()
    mask[mask != mask_body_index_list[body_mask_index]] = 0
    mask[mask == mask_body_index_list[body_mask_index]] = 1

    return mask


def get_clip_img(cv_img,
                 mask,
                 is_rgb=True,
                 patch_width=64,
                 patch_height=64,
                 on_boundary=False,
                 scale=1,
                 rot_rad=0,
                 bbox_scale=1):
    top_left, bottom_right = mask2bbox(np.repeat(mask[:, :, None], 3, axis=2))

    # The rgb image
    warped_rgb, bbox2patch = get_bbox_cropped_image_raw(
        cv_img, is_rgb,
        bbox_topleft=top_left, bbox_bottomright=bottom_right,
        patch_width=patch_width, patch_height=patch_height,
        on_boundary=on_boundary,
        bbox_scale=bbox_scale,
        scale=scale, rot_rad=rot_rad)

    return top_left, bottom_right, warped_rgb