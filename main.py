# TODO: Create a diffuser model that will be trained in all of ARC data ever existed
# TODO: Update the dataset so that it returns only the specific examples for each task
# TODO: Train the diffuser and save the model
# TODO: Create a transfer-learning loop that will train the diffuser on each task
# TODO: Create the Relational network based on the Learning to Compare: Relation Network for Few-Shot Learning paper.
# TODO: Create a training function for the Relational network and it also might need a new dataset too.
# TODO: Create a training function for the Refinement model.
from pre_train.train_gan import TrainGAN
from pre_train.train_relational import TrainRelational
from pre_train.train_refinement import TrainRefinement



if __name__ == "__main__":

   MODELS_DIR = "./trained_models"
   # cycleGAN training
   #gan_trainer = TrainGAN()
   #gan_trainer.initialize()
   #gan_trainer.run()
   #gan_trainer.evaluate()
   #gan_trainer.save(MODELS_DIR)

   #relational_trainer = TrainRelational()
   #relational_trainer.initialize()
   #relational_trainer.run()
   #relational_trainer.evaluate()
   #relational_trainer.save(MODELS_DIR)

   refinement_trainer = TrainRefinement()
   refinement_trainer.run()
