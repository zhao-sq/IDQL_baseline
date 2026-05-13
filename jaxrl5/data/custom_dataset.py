import gym
import numpy as np
from jaxrl5.data.dataset import Dataset
import sys
import pickle
import traceback

def inplace_compact_first_axis(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    assert arr.shape[0] == mask.shape[0], \
        f"mask len {mask.shape[0]} != arr.shape[0] {arr.shape[0]}"
    n = arr.shape[0]
    write = 0
    for read in range(n):
        if mask[read]:
            if write != read:
                arr[write] = arr[read]
            write += 1
    return arr[:write]

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

class CustomDataset(Dataset):
    def __init__(self, task):
        if task == 'pick_and_lift':
            zarr_path = '/media/msc-auto/HDD/szhao/LeveFD/pkl_data/pick_and_place_same_color_all.pkl'
        elif task == 'put_groceries_in_cupboard':
            zarr_path = '/media/msc-auto/HDD/szhao/LeveFD/pkl_data/put_groceries_in_cupboard_all.pkl'
        elif task == 'put_item_in_drawer':
            zarr_path = '/media/msc-auto/HDD/szhao/LeveFD/pkl_data/put_item_in_drawer_all_overhead.pkl'
        else:
            raise NotImplementedError
        try:
            with open(zarr_path, "rb") as f:
                root_data_original = NumpyCompatUnpickler(f).load()
                # with open(zarr_path, "rb") as f:
                #     root_data_original = pickle.load(f)
            print()
        except Exception as e:
            print("!! ERROR in pickle.load:", repr(e))
            traceback.print_exc()
            sys.exit(1)
        # self.delta_action = delta_action
        # if smooth_label:
        #     self.label_key = 'smooth_label'
        # else:
        #     self.label_key = 'label'
        self.img_key = ['front_rgb']
        # self.img_key = ['overhead_rgb','front_rgb']
        # self.img_key = ['left_shoulder_rgb', 'front_rgb']
        # self.img_key = ['overhead_rgb','left_shoulder_rgb']
        # if loss_type == 'contrastive':
        #     include_contrastive = True
        # else:
        #     include_contrastive = False
        img_task = False
        # dinofea_path = '/home/msc-auto/szhao/IDQL/dinov3_' + task + '_global_features.npy' 
        dinofea_path = '/home/msc-auto/szhao/IDQL/dinov2_' + task + '_patch_tokens.npy' 
        root_data = self.define_action_type(root_data_original, dinofea_path, img_task=img_task)
        post_process_data = self.post_process(root_data, img_task=img_task)

        super().__init__(post_process_data)

    def get_obs_dim(self):
        """Return the last observation dimension for vector-observation models."""
        return self.dataset_dict['observations'].shape[-1]

    def get_obs_shape(self):
        """Return the full per-sample observation shape for token-observation models."""
        return self.dataset_dict['observations'].shape[1:]

    def get_action_dim(self):
        """Return the action dimension."""
        return self.dataset_dict['actions'].shape[-1]

    def action_normalizer(self, x):
        x = np.asarray(x)

        min_v = x.min(axis=0)   # shape (8,)
        max_v = x.max(axis=0)   # shape (8,)

        scale = 2.0 / (max_v - min_v + 1e-8)
        offset = -1.0 - min_v * scale

        y = x * scale + offset
        return y

    def post_process(self, root, img_task=False):
        if not img_task:
            root['observations'] = root['data'].pop('dinov3')
            root['next_observations'] = root['data'].pop('next_dinov3')
            root['actions'] = self.action_normalizer(root['data'].pop('action'))
            label = root['data']['label']
            # epi_ends = np.concatenate(([0], root['meta']['episode_ends']))
            # rewards = np.zeros(epi_ends[-1],)
            # for i in range(len(epi_ends)-1):
            #     rewards[epi_ends[i]:epi_ends[i+1]] = 1-label[epi_ends[i]]
            rewards = 1-label
            print(rewards)
            root['rewards'] = rewards
            del root['data']['label']
            episode_ends = root['meta']['episode_ends']
            del root['meta']
            root['masks'] = np.ones(episode_ends[-1])
            dones_array = np.zeros(episode_ends[-1], dtype=bool)
            dones_array[episode_ends-1] = True
            root['dones'] = dones_array
            del root['data']
        else:
            raise NotImplementedError
        print('finish building dataset setting')
        return root

    def define_action_type(self, root, dinofea_path, action_mode='joint_pos', img_task=False):
        if action_mode == 'gripper':
            index = [root['meta']['low_dim_indice']['gripper_pose_euler'], root['meta']['low_dim_indice']['gripper_open']]
        elif action_mode == 'joint_pos':
            index = [root['meta']['low_dim_indice']['joint_positions'], root['meta']['low_dim_indice']['gripper_open']]
        elif action_mode == 'joint_vel':
            index = [root['meta']['low_dim_indice']['joint_velocities'], root['meta']['low_dim_indice']['gripper_open']]
        else:
            return NotImplementedError
        # make actions
        # root['data']['action'] = np.concatenate([root['data']['low_dim_obs'][:, index[0][0]:index[0][1]], \
        #                                          root['data']['low_dim_obs'][:, index[1][0]:index[1][1]]], axis=1)
        episodes_ends = root['meta']['episode_ends']
        obs_mask = np.ones(episodes_ends[-1], dtype=int)
        obs_mask[episodes_ends-1] = int(0)
        episodes_starts = np.concatenate((np.array([0], dtype=int), episodes_ends[:-1]))
        action_mask = np.ones(episodes_ends[-1])
        action_mask[episodes_starts] = 0
        obs_key_list = list(root['data'].keys())
        offset = np.arange(1, len(episodes_ends) + 1)
        new_episode_end = episodes_ends - offset

        root['data']['action'] = np.concatenate([root['data']['low_dim_obs'][:, index[0][0]:index[0][1]], \
                                                root['data']['low_dim_obs'][:, index[1][0]:index[1][1]]], axis=1)[action_mask.astype(bool)]   
        
        for key in obs_key_list:
            # root['data'][key] = root['data'][key][obs_mask.astype(bool)]
            if not img_task:
                if 'rgb' not in key:
                    arr_new = inplace_compact_first_axis(root['data'][key].copy(), obs_mask.astype(bool))
                    root['data'][key] = arr_new
                    assert root['data'][key].shape[0] == new_episode_end[-1]  
                else:
                    del root['data'][key] 
            else:
                arr_new = inplace_compact_first_axis(root['data'][key].copy(), obs_mask.astype(bool))
                root['data'][key] = arr_new
                assert root['data'][key].shape[0] == new_episode_end[-1]     
        if not img_task:
            dinov3_feature_info = np.load(dinofea_path)
            fea = inplace_compact_first_axis(dinov3_feature_info.copy(), obs_mask.astype(bool))
            assert fea.shape[0] == new_episode_end[-1]
            root['data']['dinov3'] = fea
            next_fea = inplace_compact_first_axis(dinov3_feature_info.copy(), action_mask.astype(bool))
            assert next_fea.shape[0] == new_episode_end[-1]
            root['data']['next_dinov3'] = next_fea
            
        assert root['data']['action'].shape[0] == new_episode_end[-1]
        root['meta']['episode_ends'] = new_episode_end
        return root
