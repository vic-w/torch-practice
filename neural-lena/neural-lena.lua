require 'image'
require 'torch'
require 'nn'
require 'qtwidget'
require 'optim'
require 'cutorch'
print(  cutorch.getDeviceProperties(cutorch.getDevice()) )
require 'cunn'

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
--l=image.load('test.jpg')
--l=l[{{1},{},{}}]
window=qtwidget.newwindow(512,512)
image.display({image=l,win=window})
 
model = nn.Sequential()
layer1 = nn.Linear(2,300)
model:add(layer1)
model:add(nn.ReLU())
layer2 = nn.Linear(300,300)
model:add(layer2)
model:add(nn.ReLU())
layer3 = nn.Linear(300,300)
model:add(layer3)
model:add(nn.ReLU())
--layer4 = nn.Linear(300,300)
--model:add(layer4)
--model:add(nn.ReLU())
layer5 = nn.Linear(300,3)
model:add(layer5)

model:cuda()

--model:add(nn.Sigmoid())
--model:add(nn.Linear(20,3))
criterion = nn.MSECriterion()
criterion:cuda()
x, dl_dx = model:getParameters()

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}



inputs={}
for yy=1,512 do
 for xx=1,512 do
  table.insert(inputs,{xx/1000,yy/1000})
 end
end
inputs = torch.Tensor(inputs)
targets = l:reshape(3,512*512):t()
inputs = inputs:cuda()
targets = targets:cuda()

local feval = function(x_new)
    -- reset data

   if x ~= x_new then x:copy(x_new) end
    dl_dx:zero()

    -- perform mini-batch gradient descent
    local output = model:forward(inputs)
    local loss = criterion:forward(output, targets)
    model:backward(inputs, criterion:backward(model.output, targets))
    return loss, dl_dx
end


for n=1,10000000 do

 _, fs = optim.adagrad(feval, x, sgd_params)

 collectgarbage()
 --print('layer1 w', layer1.weight)
 --print('layer1 b', layer1.bias)
 --print('layer2 w', layer2.weight)
 --print('layer2 b', layer2.bias)
 print('err', fs[1])
 output = model:forward(inputs):t():reshape(3,512,512)
 image.display({image=output,win=window})
 image.save('output.png', output)
end

