require 'mattorch'
display = require 'display' -- th -ldisplay.start
--https://github.com/szym/display



resultdir 	= './data/'






function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function reload_experiments(  )
	if file_exists('experiment_definitions.dat') then
		experiments = torch.load('experiment_definitions.dat')
	end
end



function experiments_already_performed( experimentID )
	expdir = resultdir .. experimentID .. '/'
	if not file_exists(expdir) then
		return 0
	end
	expRep = 0
	local resultname = expdir .. string.format('RESULT_%03i.mat', expRep + 1)
	while (file_exists(resultname)) do 
		expRep = expRep + 1
		resultname = expdir .. string.format('RESULT_%03i.mat', expRep + 1)
	end
	return expRep
end

function get_result_tensor(experiment, iteration)
	expdir = resultdir .. experiment .. '/' .. string.format('RESULT_%03i.mat', iteration)
	local success, contents = pcall(mattorch.load, expdir)
	if success then
		if opt.g then
			return contents.genseq
		else
			return contents.seq
		end
	else
		print("failed to load " .. expdir)
	end
end


function main()
	local cmd = torch.CmdLine()
	cmd:option('-g',false, 'Show generalization performance')
	cmd:option('-from',1, 'Plot from experiment')
	cmd:option('-to',math.huge,'Plot until experiment')
	cmd:option('-plot','','Optional string with the experiments to plot ')
	opt = cmd:parse(arg)

	reload_experiments()
	plotexperiments = {}
	if not(opt.plot=='') then
		for indx in string.gfind(opt.plot, "%d+") do
			plotexperiments[tonumber(indx)] = true
		end
	else
		for i=opt.from, math.min(opt.to, #experiments) do
			plotexperiments[i] = true
		end
	end

	experimentplotdata = {}
	experimentlabels = {'episode'}
	local sequences
	local trueExpCount = 1
	for i,exp in pairs(experiments) do
		
		
		if plotexperiments[i] then 

			local count = experiments_already_performed(i)
			local resindex = 0
			if count > 0 then
				for c = 1,count do
					local res = get_result_tensor(i, c)
					if res then 
						resindex = resindex + 1
						if resindex == 1 then
							sequences = torch.Tensor(res:nElement(),count)
							trueExpCount = trueExpCount + 1
						end
						sequences[{{},resindex}]:copy(res)
					end
				end
				if resindex > 0 then
					sequences = sequences[{{},{1,resindex}}]
					--display.plot(torch.cat(torch.linspace(1,sequences:size(1),sequences:size(1)),torch.cat(torch.Tensor(sequences:size(1)):fill(-30),sequences,2),2), {title=exp.name})
					experimentlabels[trueExpCount] = exp.name
					for ri = 1,sequences:size(1) do
						if not(experimentplotdata[ri]) then
							experimentplotdata[ri] = {}
						end 
						experimentplotdata[ri][1] = ri
						experimentplotdata[ri][trueExpCount] = {sequences[ri]:mean(),sequences[ri]:std()}
					end
				end
				--display.plot(torch.cat(torch.linspace(1,sequences:size(1),sequences:size(1)),sequences,2), {title=exp.name}) 
				--print(torch.cat(torch.cat(torch.linspace(1,sequences:size(1),sequences:size(1)),sequences:mean(2),2),sequences:std(2),2))
				
				
			end
		end
	end
		local typestr = '' 
		if opt.g then
			typestr = ' GENERALIZATION'
		end
		display.plot(experimentplotdata, {title='Average experiment reward trajectories' .. typestr, labels=experimentlabels ,rollPeriod = '7', showRoller = 'true', errorBars = 'true'})
end

main()
