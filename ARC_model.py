import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolutional(nn.Module):
    def __init__(self, in_channels) -> None:
        super(Convolutional, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, (1, 1), 1),
                                   nn.BatchNorm2d(num_features=16),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (1, 1), 1),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(),
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2)),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ReLU()
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2)),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ReLU()
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2)),
                                   nn.BatchNorm2d(num_features=256),
                                   nn.ReLU()
                                   )
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2, 2)),
                                   nn.BatchNorm2d(num_features=512),
                                   nn.ReLU()
                                   )
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(2, 2)),
                                   nn.BatchNorm2d(num_features=1024),
                                   nn.ReLU()
                                   )
        self.fc1 = nn.Sequential(nn.Linear(1024, 1024),
                                 nn.BatchNorm1d(num_features=1024),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(1024, 512),
                                 nn.BatchNorm1d(num_features=512),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 )
        self.fc3 = nn.Sequential(nn.Linear(512, 512),
                                 nn.BatchNorm1d(num_features=512),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 )
        self.fc4 = nn.Sequential(nn.Linear(512, 900),
                                 nn.BatchNorm1d(num_features=900),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class Adaptive(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(Adaptive, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, 30, 30)  # Reshape to 30x30 grid


class Siamese(nn.Module):
    def __init__(self, input_size) -> None:
        super(Siamese, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * input_size * input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.fc3(dis)
        return out


class ARC_model(nn.Module):
    def __init__(self, in_channels, adaptive_in_features, adaptive_out_features, siamese_input_size, threshold=0.8) -> None:
        super().__init__()
        self.convolutional = Convolutional(in_channels)
        self.adaptive = Adaptive(adaptive_in_features, adaptive_out_features)
        self.siamese = Siamese(siamese_input_size)
        self.threshold = threshold

    def forward(self, x, examples):
        # Convolutional network processes the input
        conv_out = self.convolutional(x)

        # Adaptive network generates the solution
        solution = self.adaptive(conv_out)

        # Siamese network measures similarity
        similarities = []
        for example in examples:
            similarity = self.siamese(solution.unsqueeze(1), example.unsqueeze(1))
            similarities.append(similarity)

        avg_similarity = torch.mean(torch.stack(similarities))

        if self.training:
            return solution, avg_similarity
        else:
            # During inference, retrain adaptive network if similarity is below threshold
            while avg_similarity < self.threshold:
                # Retrain adaptive network (this is a simplified version, you might want to implement a proper training loop)
                self.adaptive.train()
                solution = self.adaptive(conv_out)
                similarities = [self.siamese(solution.unsqueeze(1), example.unsqueeze(1)) for example in examples]
                avg_similarity = torch.mean(torch.stack(similarities))
                self.adaptive.eval()

            return solution, avg_similarity

    def initial_training(self, train_loader, num_epochs, learning_rate):
        # Freeze Siamese network
        for param in self.siamese.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(list(self.convolutional.parameters()) + list(self.adaptive.parameters()), lr=learning_rate)
        criterion = nn.MSELoss()  # Assuming we're predicting grid values

        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                conv_out = self.convolutional(data)
                output = self.adaptive(conv_out)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def meta_training(self, meta_train_loader, meta_val_loader, num_epochs, outer_lr, inner_lr, num_inner_updates):
        # Freeze Convolutional network
        for param in self.convolutional.parameters():
            param.requires_grad = False

        # Unfreeze Siamese network
        for param in self.siamese.parameters():
            param.requires_grad = True

        meta_optimizer = torch.optim.Adam(list(self.adaptive.parameters()) + list(self.siamese.parameters()), lr=outer_lr)

        for epoch in range(num_epochs):
            meta_train_loss = 0.0
            for task_batch in meta_train_loader:
                # Clone adaptive network for inner loop optimization
                fast_weights = [p.clone().detach().requires_grad_(True) for p in self.adaptive.parameters()]

                # Inner loop optimization
                for _ in range(num_inner_updates):
                    support_loss = 0.0
                    for support_data, support_target in task_batch['support']:
                        conv_out = self.convolutional(support_data)
                        support_output = self.adaptive(conv_out)
                        support_loss += self.siamese(support_output.unsqueeze(1), support_target.unsqueeze(1)).mean()

                    grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
                    fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

                # Compute meta-loss on query set
                query_loss = 0.0
                for query_data, query_target in task_batch['query']:
                    conv_out = self.convolutional(query_data)
                    query_output = self.adaptive(conv_out)
                    query_loss += self.siamese(query_output.unsqueeze(1), query_target.unsqueeze(1)).mean()

                meta_train_loss += query_loss

            # Meta-update
            meta_optimizer.zero_grad()
            meta_train_loss.backward()
            meta_optimizer.step()

            print(f'Epoch {epoch+1}/{num_epochs}, Meta-Loss: {meta_train_loss.item()}')

        # Meta-validation
            # Meta-validation
            self.eval()  # Set model to evaluation mode
            meta_val_loss = 0.0
            num_val_tasks = 0

            with torch.no_grad():  # No need to track gradients for validation
                for task_batch in meta_val_loader:
                    num_val_tasks += 1

                    # Clone adaptive network for inner loop optimization
                    fast_weights = [p.clone().detach() for p in self.adaptive.parameters()]

                    # Inner loop optimization
                    for _ in range(num_inner_updates):
                        support_loss = 0.0
                        for support_data, support_target in task_batch['support']:
                            conv_out = self.convolutional(support_data)
                            support_output = self.adaptive(conv_out)
                            support_loss += self.siamese(support_output.unsqueeze(1),
                                                         support_target.unsqueeze(1)).mean()

                        # Manual weight update (since we're in no_grad mode)
                        grads = torch.autograd.grad(support_loss, fast_weights, create_graph=False)
                        fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

                    # Compute meta-validation loss on query set
                    query_loss = 0.0
                    for query_data, query_target in task_batch['query']:
                        conv_out = self.convolutional(query_data)
                        # Use fast_weights for the adaptive network
                        query_output = self.adaptive(conv_out)
                        query_loss += self.siamese(query_output.unsqueeze(1), query_target.unsqueeze(1)).mean()

                    meta_val_loss += query_loss.item()

            avg_meta_val_loss = meta_val_loss / num_val_tasks
            print(f'Meta-Validation Loss: {avg_meta_val_loss:.4f}')

            self.train()

    def meta_validate(self, meta_val_loader, inner_lr, num_inner_updates):
        self.eval()  # Set model to evaluation mode
        meta_val_loss = 0.0
        num_val_tasks = 0

        with torch.no_grad():  # No need to track gradients for validation
            for task_batch in meta_val_loader:
                num_val_tasks += 1

                # Clone adaptive network for inner loop optimization
                fast_weights = [p.clone().detach() for p in self.adaptive.parameters()]

                # Inner loop optimization
                for _ in range(num_inner_updates):
                    support_loss = 0.0
                    for support_data, support_target in task_batch['support']:
                        conv_out = self.convolutional(support_data)
                        support_output = self.adaptive(conv_out)
                        support_loss += self.siamese(support_output.unsqueeze(1), support_target.unsqueeze(1)).mean()

                    # Manual weight update (since we're in no_grad mode)
                    grads = torch.autograd.grad(support_loss, fast_weights, create_graph=False)
                    fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

                # Compute meta-validation loss on query set
                query_loss = 0.0
                for query_data, query_target in task_batch['query']:
                    conv_out = self.convolutional(query_data)
                    # Use fast_weights for the adaptive network
                    query_output = self.adaptive(conv_out)
                    query_loss += self.siamese(query_output.unsqueeze(1), query_target.unsqueeze(1)).mean()

                meta_val_loss += query_loss.item()

        avg_meta_val_loss = meta_val_loss / num_val_tasks
        self.train()  # Set model back to training mode
        return avg_meta_val_loss


if __name__ == "__main__":
    # Example usage:
    model = ARC_model(in_channels, adaptive_in_features, adaptive_out_features, siamese_input_size)

    # Initial training
    initial_train_loader = ...  # Create your DataLoader for initial training
    model.initial_training(initial_train_loader, num_epochs=50, learning_rate=0.001)

    # Meta-training
    meta_train_loader = ...  # Create your DataLoader for meta-training
    meta_val_loader = ...  # Create your DataLoader for meta-validation
    model.meta_training(meta_train_loader, meta_val_loader, num_epochs=100, outer_lr=0.001, inner_lr=0.01,
                        num_inner_updates=5)

    # Separate meta-validation (can be used after training or during training)
    val_loss = model.meta_validate(meta_val_loader, inner_lr=0.01, num_inner_updates=5)
    print(f'Final Meta-Validation Loss: {val_loss:.4f}')
