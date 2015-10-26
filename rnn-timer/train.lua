require 'rnn'
require 'gnuplot'

batchSize = 8
rho = 100
hiddenSize = 20

-- RNN

batchLoader = require 'MinibatchLoader'
loader = batchLoader.create(batchSize)

r = nn.Recurrent(
   hiddenSize, nn.Linear(1, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(nn.Sequencer(r))
rnn:add(nn.Sequencer(nn.Linear(hiddenSize, 1)))
rnn:add(nn.Sequencer(nn.Sigmoid()))

criterion = nn.SequencerCriterion(nn.MSECriterion())


lr = 0.01
i = 1
for n=1,6000 do
   -- prepare inputs and targets
   local inputs, targets = loader:next_batch()

   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   print(i, err/rho)
   i = i + 1
   local gradOutputs = criterion:backward(outputs, targets)
   rnn:backward(inputs, gradOutputs)
   rnn:updateParameters(lr)
   rnn:zeroGradParameters()
end

inputs, targets = loader:next_batch()
outputs = rnn:forward(inputs)

x={}
y={}
for i=1,100 do
   table.insert(x,inputs[i][{1,1}])
   table.insert(y,outputs[i][{1,1}])
end

x = torch.Tensor(x)
y = torch.Tensor(y)
	
gnuplot.pngfigure('timer.png')
gnuplot.plot({x},{y})
gnuplot.plotflush()
