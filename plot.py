import numpy as np
import matplotlib.pyplot as plt

def plot():
  #plotting the losses and accuracies
  x=np.linspace(0,epoch_count,epoch_count*4)
  x1=np.arange(0,epoch_count)

  plt.figure(figsize=(10,7))
  plt.plot(
    x,foo1,'g-',label='training loss'
  )
  plt.plot(
    x1,foo3,'b-',label='validation loss'
  )
  plt.xlabel("Epoch count")
  plt.ylabel("Magnitude")
  plt.title("Training and Validation Losses")
  plt.legend()
  plt.savefig('val_train_loss.png',dpi=300)
  plt.show()

  plt.figure(figsize=(10,7))
  plt.plot(
    x,foo2,'r-',label='training accuracy'
  )
  plt.plot(
    x1,foo4,'b-',label='validation accuracy'
  )
  plt.xlabel("Epoch count") 
  plt.ylabel("Magnitude")
  plt.title("Training and Validation Accuracy")
  plt.legend()
  plt.savefig('val_train_acccuracy.png',dpi=300)
  plt.show()
  
  return
