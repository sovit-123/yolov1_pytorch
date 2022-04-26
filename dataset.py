import torch.utils.data as data
import torch
import cv2

class DetectionDataset(data.Dataset):
    def __init__(self, image_size, file, train, transform, S, C, B):
        self.image_size = image_size
        self.file = file
        self.train = train
        self.transform = transform
        self.S = S
        self.C = C
        self.B = B

        self.image_names = []
        self.bboxes = []
        self.labels = []

        with open(file) as f:
            lines = f.readlines()

        # Iterate over the image paths and targets, and,
        # extract the required info.
        for i, line in enumerate(lines):
            # Split the file path and annotations according to white space.
            split_line = line.strip().split()
            # Index 0 is the file path
            self.image_names.append(split_line[0])
            # Except the image path, all other are bbox info in the format,
            # x1, y1, x2, y2, labe, x1, y1, x2, y2, label, ...
            num_bboxes = (len(split_line) - 1) // 5
            bbox = []
            label = []
            for j in range(num_bboxes):
                x1 = float(split_line[1+5*j])
                y1 = float(split_line[2+5*j])
                x2 = float(split_line[3+5*j])
                y2 = float(split_line[4+5*j])
                class_num = split_line[5+5*j]
                bbox.append([x1, y1, x2, y2])
                label.append(int(class_num))
            self.bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.long))
        self.num_samples = len(self.bboxes)

    def __len__(self):
        return self.num_samples

    def encoder(self, bboxes, labels):
        """
        :param boxes: Bounding box tensor as [[x1, y1, x2, y2], [], ...]
        :param labels: Label tensor

        Returns:
            target: Shape of grid_sizexgrid_sizex30. In paper (7x7x30)
        """
        label_matrix = torch.zeros(self.S, self.S, self.C + 5 * self.B)
        for i, box in enumerate(bboxes):
            class_label = labels[i]
            width, height = box[2:] - box[:2]
            # Get mid points x and y.
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object.
                label_matrix[i, j, self.C] = 1
                # Box coordinates.
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                # Set one hot encoding for class_label.
                label_matrix[i, j, class_label] = 1
        # Final label matrix has shape 7x7x30 if S=7, C=20, and B=2.
        return label_matrix

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = cv2.imread(image_name)
        orig_height, orig_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        bboxes = self.bboxes[index].clone()
        labels = self.labels[index].clone()

        if self.train:
            # Image augmentation here
            pass
        
        # Normalize the bounding boxes as per the original image dimensions.
        bboxes /= torch.tensor([orig_width, orig_height, orig_width, orig_height])
        # bboxes = bboxes.expand_as(bboxes)
        target = self.encoder(bboxes, labels)
        if self.transform:
            image = self.transform(image)
        return image, target

if __name__ == '__main__':
    import transforms
    transform = transforms.get_tensor_transform()
    train_dataset = DetectionDataset(
        image_size=448, file='2007_train_labels.txt', 
        train=True, transform=transform, S=7, C=20, B=2
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=4,
        shuffle=True, num_workers=4
    )

    def get_mean_and_std(dataloader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in dataloader:
            channels_sum += torch.mean(data, dim=[0,2,3])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1
        
        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std
    
    mean, std = get_mean_and_std(train_loader)
    print(f"Mean: {mean} ::: Std: {std}")