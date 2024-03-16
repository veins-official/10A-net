import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torch import nn
import torch, os
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path1, path2):
        super().__init__()

        self.path1 = path1
        self.path2 = path2

        self.list1 = os.listdir(path1)
        self.list2 = os.listdir(path2)
    

    def __len__(self):
        return len(self.list1) + len(self.list2)
    

    def __getitem__(self, index):
        if index < len(self.list1):
            class_id = 0
            img_path = os.path.join(self.path1, self.list1[index])
        else:
            class_id = 1
            index -= len(self.list1)
            img_path = os.path.join(self.path2, self.list2[index])

        image = Image.open(img_path)
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]) # создаём единый вид для изображений
        image_tensor = transform(image) # приводим излбражение к единому виду
        t_class_id = torch.tensor(class_id)

        return { "image": image_tensor, "label": t_class_id }


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(64 * 64 * 3, 128)
        self.linear2 = nn.Linear(128, 2)
        self.act = nn.ReLU()


    def forward(self, x):
        out = self.flat(x)      # приводим тензоры к одному виду
        out = self.linear1(out) # 1-ый слой
        out = self.act(out)     # функция активации
        out = self.linear2(out) # 2-ой слой

        return out


def show_accuracy(accuracy_path):
    with open(accuracy_path, "r") as file: accuracy = [float(line) for line in file.read().split()]

    plt.title("ACCURACY PROGRESS")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")

    plt.plot(range(len(accuracy)), accuracy)
    plt.show()


def accuracy(pred, label):
    answer = F.softmax(pred.detach(), -1).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


if __name__ == "__main__":
    model_path, accuracy_path, test_path = os.path.join(".", "model"), os.path.join(".", "accuracy"), os.path.join(".", "test") # задаём пути к папкам


    OLD_MODEL = True # будет ли использоваться прошлая модель
    TRAIN     = True # нужно ли её тренировать
    TEST      = True # проводить ли тест модели
    EPOCHS    = 50   # кол-во эпох обучения модели


    model = Network()
    
    # загрузка модели
    if OLD_MODEL and os.path.exists(model_path):
        model = torch.load(model_path)
        model.eval()
    # загрузка модели


    if TRAIN:
        dataset = Dataset(os.path.join(".", "train", "cats"), os.path.join(".", "train", "dogs"))
        
        batch_size = 16
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        accuracy_list = []

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(EPOCHS):
            accuracy_val = 0
            iters_count = 0

            for sample in train_loader:
                img, label = sample["image"], sample["label"]
                optimizer.zero_grad() # обнуление градиентов

                label = F.one_hot(label, 2).float() # предпологаемый output
                pred = model(img) # действительный output

                loss = loss_fn(pred, label) # потери

                loss.backward() # обратное распространение ошибки

                accuracy_val += accuracy(pred, label)
                iters_count += 1
                optimizer.step()

            print(f"EPOCH: { epoch }, ACCURACY: { "{:.2f}".format((accuracy_val / iters_count) * 100) }%")
            accuracy_list.append((accuracy_val / iters_count) * 100)

        with open(accuracy_path, "a" if OLD_MODEL else "w") as file: file.write("\n".join([str(num) for num in accuracy_list]) + "\n")
        torch.save(model, model_path)

        show_accuracy(accuracy_path)
    

    if TEST:
        columns, rows = math.ceil(math.sqrt(len(os.listdir(test_path)))), round(math.sqrt(len(os.listdir(test_path))))
        fig = plt.figure(figsize=(5, 5))

        for index, path in enumerate(os.listdir(test_path)):
            image = Image.open(os.path.join(test_path, path))
            transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
            image_tensor = transform(image)
            output = model(image_tensor[None])

            fig.add_subplot(rows, columns, index + 1)

            plt.imshow(image)
            plt.axis("off")
            plt.title("собачка" if F.softmax(output.detach(), -1).numpy().argmax(1)[0] else "котик")
        
        plt.show()
