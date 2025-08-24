import os
import os.path as osp
from .bases import BaseImageDataset

class NDTwin(BaseImageDataset):
    """
    NDTwin dataset for verification task
    """
    dataset_dir = 'ndtwin'

    def __init__(self, root='your_data_root', verbose=True, **kwargs):
        super(NDTwin, self).__init__()
        self.data_root = root  # This should be the root directory
        
        # Load train and test splits from txt files
        train_txt = osp.join(root, 'train.txt')
        test_txt = osp.join(root, 'test.txt')
        
        self._check_before_run(train_txt, test_txt)
        
        # Process training data (for Re-ID style training)
        train = self._process_train_data(train_txt, relabel=True)
        
        # Process test data (for verification evaluation)
        test_pairs = self._process_test_data(test_txt)
        
        if verbose:
            print("=> NDTwin loaded")
            self.print_dataset_statistics(train, test_pairs)
        
        self.train = train
        self.test_pairs = test_pairs
        
        # For compatibility with existing code, create dummy query and gallery
        self.query = train[:len(train)//2]  # Use first half as query
        self.gallery = train[len(train)//2:]  # Use second half as gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self, train_txt, test_txt):
        """Check if all files are available"""
        if not osp.exists(train_txt):
            raise RuntimeError("'{}' is not available".format(train_txt))
        if not osp.exists(test_txt):
            raise RuntimeError("'{}' is not available".format(test_txt))

    def _process_train_data(self, train_txt, relabel=False):
        """Process training data for Re-ID style training"""
        with open(train_txt, 'r') as f:
            lines = f.readlines()
        
        # Extract unique identities
        pid_container = set()
        for line in lines:
            image_name = line.strip()
            pid = image_name[:5]  # First 5 characters as identity
            pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for line in lines:
            image_name = line.strip()
            pid = image_name[:5]
            img_path = osp.join(self.data_root, pid, image_name)
            
            if relabel:
                pid = pid2label[pid]
            
            # Set all camera IDs to 0 since you don't have camera info
            camid = 0
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_test_data(self, test_txt):
        """Process test data for verification pairs"""
        with open(test_txt, 'r') as f:
            lines = f.readlines()
        
        test_pairs = []
        for line in lines:
            # Expected format: "img1_name img2_name label"
            parts = line.strip().split()
            if len(parts) == 3:
                img1_name, img2_name, label = parts
                label = int(label)
                
                # Construct full paths
                img1_path = osp.join(self.data_root, img1_name[:5], img1_name)
                img2_path = osp.join(self.data_root, img2_name[:5], img2_name)
                
                test_pairs.append((img1_path, img2_path, label))

        return test_pairs

    def print_dataset_statistics(self, train, test_pairs):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_test_pairs = len(test_pairs)
        num_positive_pairs = sum(1 for _, _, label in test_pairs if label == 1)
        num_negative_pairs = num_test_pairs - num_positive_pairs

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  ----------------------------------------")
        print("  test pairs | positive | negative | total")
        print("  ----------------------------------------")
        print("  {:10d} | {:8d} | {:8d} | {:5d}".format(num_test_pairs, num_positive_pairs, num_negative_pairs, num_test_pairs))
        print("  ----------------------------------------")