resultdir 	= './data/'
LOCAL_DATA_DIR = resultdir
REMOTE_DATA_DIR =  's3://timdebruin/experiencereplay/jmlr_data/'
experimentrunfile = 'runsim.lua'

local cmd = torch.CmdLine()
cmd:option('-name','lab', 'ID of this PC (lab laptop desktop)')
opt = cmd:parse(arg)

function wait_for_result(filename)
	while not ( file_exists(filename .. 'completed')) do
		os.execute("sleep 3")
		check_command_messages()
	end
	os.execute('rm ' .. filename .. 'completed') 
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function check_command_messages()
	local stopped = false
	while file_exists('stop') or file_exists('stop.txt') do
		if not stopped then 
			stopped = true
			report_status("Noticed a stop command file, will resume when it is removed.")
		end
		os.execute("sleep 60")
	end
	if stopped then
		report_status("Resuming operations")
	end
	if file_exists('reload_experiments-' .. opt.name) then
		reload_experiments()
	end
end

function report_status(status,nodate)
	if not nodate then 
		status = os.date("%A , %X , " .. opt.name .. ": ") .. status 
	end
	print(status)
	local f=io.open(resultdir .. 'automated_experiment_status.txt',"a")
	f:write(status .. '\n')
	f:close()
end

function reload_experiments(  )
	if file_exists('experiment_definitions.dat') then
		experiments = torch.load('experiment_definitions.dat')
		if file_exists('reload_experiments-' .. opt.name) then 
			os.execute('rm reload_experiments-' .. opt.name)
		end
		report_status("Experiment definitions reloaded. Open experiments:")
		for i,e in ipairs(experiments) do
			if e.execute then
				report_status('Experiment ' .. i .. ' : ' .. e.name)
			end
		end
	else
		report_status('WARNING: File experiment_definitions.dat not found. Fix this and remove the stop file to resume.')
		torch.save('stop',{})
		check_command_messages()
	end
end

function perform_experiment( experimentID )
	expdir = resultdir .. experimentID .. '/'
	if not file_exists(expdir) then
		os.execute('mkdir ' .. expdir)
	end
	os.execute('cp  ' .. experimentrunfile  .. ' ' .. expdir .. 'backup_' .. experimentrunfile )
	local namenotfound = true
	local expRep = 1		
	local resultname = expdir .. string.format('RESULT_%03i.mat', expRep)
	while (file_exists(resultname)) do 
		expRep = expRep + 1
		resultname = expdir .. string.format('RESULT_%03i.mat', expRep)
	end
	
	torch.save(resultname,{})
	send_data()
	report_status('Reserved ' .. resultname)
	local command = 'gnome-terminal -e "th ' .. experimentrunfile .. ' -resultfile \'' .. resultname .. '\' ' .. experiments[experimentID].paramstring .. '"'
	if opt.name == 'lab' or opt.name == 'laptop' then
		command = 'th ' .. experimentrunfile .. ' -resultfile \'' .. resultname .. '\' ' .. experiments[experimentID].paramstring 
	end	
	report_status('Attempting experiment: ' .. experimentID .. '(' ..experiments[experimentID].name .. ') iteration ' .. expRep .. ' with the following command:' )
	report_status(command)
		
	os.execute(command)
	wait_for_result(resultname) 
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

function first_experiment_with_least_results()
	local leastRuns = math.huge
	local leastRunsID
	for experiment = 1,#experiments do
		if experiments[experiment].execute then
			local runs = experiments_already_performed(experiment)
			if runs < leastRuns then
				leastRuns 	= runs
				leastRunsID = experiment 
			end
		end
	end
	return leastRunsID
end

function get_name(	)
	local f = io.popen("/bin/hostname")
	local hostname = f:read("*a") or ''
	f:close()
	hostname = string.gsub(hostname, "\n$" , "")
	return hostname
end

function load_data()
	os.execute('aws s3 sync ' .. REMOTE_DATA_DIR .. ' ' .. LOCAL_DATA_DIR)
end

function send_data()
	os.execute('aws s3 sync ' .. LOCAL_DATA_DIR .. ' ' .. REMOTE_DATA_DIR)
end

function main()
	--NAME = get_name()
	assert((opt.name == 'desktop' or opt.name == 'laptop' or opt.name == 'lab'),'specify -name as lab desktop or laptop')	
	load_data()
	report_status(os.date("======  %A %d %B %Y ======"),true)
	report_status('Starting automated experiment cycle')
	reload_experiments()
	
	while true do
		os.execute('git pull')
		check_command_messages()
		while not experiments do
			reload_experiments()
		end
		load_data()
		local nextExperiment = first_experiment_with_least_results()
		if nextExperiment then
			perform_experiment(nextExperiment)
			if not(opt.name=='lab') then
				os.execute('th converttomat.lua')
			end
			send_data()
			--os.execute('sleep 60') -- if anything funny happens in the loop (eg with creating folders, at least it only happens once a minute...)
		else
			os.execute('sleep 30') -- wait for experiments to be defined	
		end	
	end
end

main()
