require 'image'
require 'torch'
require 'nn'

function gradientUpdate(model, x, y, learningRate)
   local criterion = nn.MSECriterion()
   local prediction = model:forward(x)
   local err = criterion:forward(prediction, y)
   model:zeroGradParameters()
   local gradient = criterion:backward(prediction, y)
   model:backward(x, gradient)
   model:updateParameters(learningRate)
   return err
end

l=image.lena()
image.window()
image.display(l)
 
model = nn.Sequential()
layer1 = nn.Linear(2,20)
model:add(layer1)
model:add(nn.Sigmoid())
--model:add(nn.Linear(20,20))
--model:add(nn.Sigmoid())
model:add(nn.Linear(20,3))

inputs={}
for y=1,512 do
 for x=1,512 do
  table.insert(inputs,{x/1000,y/1000})
 end
end
inputs = torch.Tensor(inputs)
targets = l:reshape(3,512*512):t()


for n=1,100 do
for n2=1,10 do
 err = gradientUpdate(model, inputs, targets, 0.01)
end
 print(err)
 output = model:forward(inputs):t():reshape(3,512,512)
 image.display(output)
end

