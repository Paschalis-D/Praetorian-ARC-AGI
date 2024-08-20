# TODO: Create a diffuser model that will be trained in all of ARC data ever existed
# TODO: Update the dataset so that it returns only the specific examples for each task
# TODO: Train the diffuser and save the model
# TODO: Create a transfer-learning loop that will train the diffuser on each task
# TODO: Create the Relational network based on the Learning to Compare: Relation Network for Few-Shot Learning paper.
# TODO: Create a training function for the Relational network and it also might need a new dataset too.
# TODO: Create a training function for the Refinement model.
from pre_train.train_gan import TrainGAN



if __name__ == "__main__":
    # cycleGAN training
    gan_trainer = TrainGAN()

    # Initialize models, dataloaders etc..
    gan_trainer.initialize()

    # Train the GAN generators
    gan_trainer.run()

    # Evaluate
    gan_trainer.evaluate()
