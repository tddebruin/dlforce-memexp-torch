require 'mattorch' 
resultdir 	= './data/'

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function experiments_already_performed( experimentID )
	expdir = resultdir .. experimentID .. '/'
	if not file_exists(expdir) then
		return 0
	end
	expRep = 1
	local resultname = expdir .. string.format('RESULT_%03i.mat', expRep + 1)
	while (file_exists(resultname)) do 
		expRep = expRep + 1
		resultname = expdir .. string.format('RESULT_%03i.mat', expRep + 1)
	end
	return expRep
end

experiments = torch.load('experiment_definitions.dat')

for i=1,#experiments do
	local numex = experiments_already_performed(i)
	for j=1,numex do
		local infilename = expdir .. string.format('RESULT_%03i.mat-t7', j)
		if file_exists(infilename) then
			local temp = torch.load(infilename)
			local outfilename = expdir .. string.format('RESULT_%03i.mat', j)
			mattorch.save(outfilename, temp)
			os.execute('rm ' .. infilename)
		end
	end
end

