import numpy as np


class RandomMaskingGenerator2D:
    def __init__(self, input_size, mask_ratio):
        self.height, self.width = input_size
        self.num_patches =  self.height * self.width
        self.num_masks = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_masks
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_masks), # 0: for unmasked
            np.ones(self.num_masks), # 1: for masked
        ])
        np.random.shuffle(mask) # in-place
        return mask


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask


class TubeWindowMaskingGenerator:
    
    def __init__(self, input_size, mask_ratio, win_size, apply_symmetry=None):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

        assert self.width == win_size[1] and win_size[1] % 2 == 0, "Error: window width must be equal to input width and be even if apply windown attn and face symmetrical masking."

        assert self.height % win_size[0] == 0 and self.width % win_size[1] == 0
        self.spatial_part_size = (self.height // win_size[0], self.width // win_size[1])
        self.num_wins_per_frame = self.spatial_part_size[0] * self.spatial_part_size[1]
        self.num_patches_per_win = win_size[0] * win_size[1]
        self.num_masks_per_win = int(mask_ratio * self.num_patches_per_win)
        self.num_unmasks_per_win = self.num_patches_per_win - self.num_masks_per_win
        self.apply_symmetry = apply_symmetry
        if apply_symmetry is not None:
            assert apply_symmetry in ['global', 'local']
            assert mask_ratio > 0.5
            self.win_width_half = (win_size[1] // 2)
            self.num_patches_per_win_half = win_size[0] * self.win_width_half
            self.win_size_half = (win_size[0], self.win_width_half)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = []
        if self.apply_symmetry is None:
            for i in range(self.num_wins_per_frame):
                mask_per_win = [0] * (self.num_patches_per_win - self.num_masks_per_win) + [1] * self.num_masks_per_win
                np.random.shuffle(mask_per_win)
                mask_per_frame.extend(mask_per_win)
        elif self.apply_symmetry == 'global':
            left = np.random.rand() < 0.5
            mask_per_frame = []
            for i in range(self.num_wins_per_frame):
                mask_per_win_half = [0] * self.num_unmasks_per_win + [1] * (self.num_patches_per_win_half - self.num_unmasks_per_win)
                np.random.shuffle(mask_per_win_half)
                mask_per_win = []
                for i in range(self.win_size_half[0]):
                    if left:
                        mask_per_win.extend([mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] + [1] * self.win_size_half[1]])
                    else:
                        mask_per_win.extend([[1] * self.win_size_half[1] + mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]]])
                # if left:
                #     # mask_per_win = np.hstack([np.array(mask_per_win_half).reshape(self.win_size_half), np.ones(self.win_size_half)])
                #     mask_per_win = [mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] + [1] * self.win_size_half[1] for i in self.win_size_half[0]]
                # else:
                #     # mask_per_win = np.hstack([np.ones(self.win_size_half), np.array(mask_per_win_half).reshape(self.win_size_half)])
                #     mask_per_win = [[1] * self.win_size_half[1] + mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] for i in self.win_size_half[0]]
                # mask_per_frame.append(mask_per_win.flatten())
                mask_per_frame.extend(mask_per_win)
            # mask_per_frame = np.hstack(mask_per_frame)
        else: # local
            mask_per_frame = []
            for i in range(self.num_wins_per_frame):
                mask_per_win_half = [0] * self.num_unmasks_per_win + [1] * (self.num_patches_per_win_half - self.num_unmasks_per_win)
                np.random.shuffle(mask_per_win_half)
                left = np.random.rand() < 0.5
                mask_per_win = []
                for i in range(self.win_size_half[0]):
                    if left:
                        mask_per_win.extend([mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] + [1] * self.win_size_half[1]])
                    else:
                        mask_per_win.extend([[1] * self.win_size_half[1] + mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]]])
                mask_per_frame.extend(mask_per_win)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask
    
class MotionAwareMaskingGenerator:
    """
    Generate a mask based on motion detection using bidirectional frame differencing.

    - Splits each frame into patches of size (patch_h, patch_w).
    - Computes motion score per patch by:
        diff1 = abs(frame_t   - frame_{t-1})
        diff2 = abs(frame_{t+1} - frame_t)
        motion_map = (diff1 + diff2) / 2
        patch_score = average motion_map within each patch.
    - Uses a motion_threshold to classify patches as moving or static.
    - Applies high_mask_ratio to static patches and low_mask_ratio to moving patches.
    """
    def __init__(self,
                 frames: int,
                 image_size: tuple,
                 patch_size: tuple,
                 mask_ratio: float,
                 motion_threshold: float = 10.0,
                 static_mask_ratio: float = 0.9,
                 motion_mask_ratio: float = 0.1):
        """
        Args:
            frames: number of frames in clip (T)
            image_size: (H, W)
            patch_size: (patch_h, patch_w)
            motion_threshold: pixel-difference threshold to flag motion
            static_mask_ratio: mask ratio for static patches (0-1)
            motion_mask_ratio: mask ratio for moving patches (0-1)
        """
        self.frames = frames
        self.H, self.W = image_size
        self.ph, self.pw = patch_size
        assert self.H % self.ph == 0 and self.W % self.pw == 0, \
            "Image dimensions must be divisible by patch size"
        # number of patches per frame
        self.n_h = self.H // self.ph
        self.n_w = self.W // self.pw
        # self.n_patches_per_frame = self.n_h * self.n_w
        self.total_patches = self.frames * self.n_h * self.n_w
        
        self.mask_ratio = mask_ratio
        self.motion_threshold = motion_threshold
        self.static_mask_ratio = static_mask_ratio
        self.motion_mask_ratio = motion_mask_ratio

    def __repr__(self):
        return (f"MotionAwareMasking(frames={self.frames}, size={self.H}x{self.W}, "
                f"patch={self.ph}x{self.pw}, thresh={self.motion_threshold}, "
                f"static_ratio={self.static_mask_ratio}, motion_ratio={self.motion_mask_ratio})")
    
    

    def __call__(self, frames_array: np.ndarray):
        # frames_array 形状 (T_orig, H, W, C)，T_orig = self.frames * segment_size (2)
        T_orig, H, W, C = frames_array.shape
        assert T_orig % self.frames == 0, "帧数不是 self.frames 的整数倍"
        segment_size = T_orig // self.frames

        # 1) collapse 每 segment_size 帧 到 1 帧
        collapsed = frames_array.reshape(self.frames,
                                         segment_size, H, W, C).mean(axis=1)  # (self.frames,H,W,C)

        # 2) 下面就像你原先写的一样，用 collapsed 计算 motion_map
        gray = collapsed.mean(axis=-1)  # (self.frames,H,W)
        motion_map = np.zeros_like(gray)
        for t in range(self.frames):
            d1 = np.abs(gray[t] - gray[t-1]) if t>0 else 0
            d2 = np.abs(gray[t+1] - gray[t]) if t<self.frames-1 else 0
            motion_map[t] = 0.5*(d1 + d2)

        # 3) 汇聚到 patch 级
        n_h, n_w = self.H//self.ph, self.W//self.pw
        patch_scores = motion_map.reshape(self.frames,
                                          n_h, self.ph,
                                          n_w, self.pw).mean(axis=(2,4))  # (self.frames,n_h,n_w)
        flat_scores = patch_scores.flatten()  # (self.frames*n_h*n_w,)

        # 4) 加权抽样 M = mask_ratio * total_patches
        weights = np.where(flat_scores < self.motion_threshold,
                           self.static_mask_ratio,
                           self.motion_mask_ratio)
        weights = weights/weights.sum()
        M = int(self.mask_ratio * flat_scores.size)
        chosen = np.random.choice(flat_scores.size,
                                  size=M, replace=False, p=weights)
        mask = np.zeros(flat_scores.size, dtype=np.int32)
        mask[chosen] = 1
        return mask