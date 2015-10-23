require 'image'
require 'torch'
require 'nn'
require 'qtwidget'

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

--l=image.lena()
l=image.load('test.jpg')
l=l[{{1},{},{}}]
window=qtwidget.newwindow(512,512)
image.display({image=l,win=window})
 
model = nn.Sequential()
layer1 = nn.Linear(2,10)
model:add(layer1)
model:add(nn.Sigmoid())
layer2 = nn.Linear(10,1)
model:add(layer2)
--model:add(nn.Sigmoid())
--model:add(nn.Linear(20,3))

inputs={}
for y=1,512 do
 for x=1,512 do
  table.insert(inputs,{x/1000,y/1000})
 end
end
inputs = torch.Tensor(inputs)
targets = l:reshape(1,512*512):t()


for n=1,10000 do
for n2=1,100 do
 err = gradientUpdate(model, inputs, targets, 0.8)
end
 collectgarbage()
 --print('layer1 w', layer1.weight)
 --print('layer1 b', layer1.bias)
 --print('layer2 w', layer2.weight)
 --print('layer2 b', layer2.bias)
 print('err', err)
 output = model:forward(inputs):t():reshape(1,512,512)
 image.display({image=output,win=window})
end

