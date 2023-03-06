def train(model):
  #variables for tracking the losses and accuracies
  foo1=[]#train_loss
  foo2=[]#train_accuracy
  foo3=[]#valid_loss
  foo4=[]#valid_accuracy
  bestmodel={'epoch':  0,
        'model_state_dict': model.state_dict(),
        'loss':   0,
        'accuracy':  0
        }

  for epoch in range(1,epoch_count+1):
      running_loss=0.0
      running_total = 0
      running_correct = 0
      run_step = 0
   
      for i, (images, labels) in enumerate(train_loader):
        model.train()  
        
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss += loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i %  500== 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  f'train_loss: {running_loss / run_step :.3f}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            foo1.append(running_loss/run_step)
            foo2.append(100 * running_correct / running_total)
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
  return foo1,foo2,foo3,foo4
