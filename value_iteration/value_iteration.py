# Roshan Shrestha


import sys

if len(sys.argv) != 5:
	print("python value_iteration.py <environment_file> <non_terminal_reward> <gamma> <K>")
	exit(0)

environment_file = str(sys.argv[1])
non_terminal_reward = float(sys.argv[2])
gamma = float(sys.argv[3])
K = int(sys.argv[4])

environment = []

def create_state(terminalReward, terminalState, blocked):
	state = {"reward": terminalReward, "utility": 0,\
		"isTerminalState": terminalState, "isBlocked": blocked}
	return state

import csv
with open(environment_file, newline='') as csvfile:
	line = csv.reader(csvfile, delimiter=',', quotechar='|')
	environment_row = []
	for row in line:
		environment_row = []
		for element in row:
			environment_row.append(element)			
		environment.append(environment_row)

states = []
for i in range(len(environment)+2):
	row = []
	for j in range(len(environment[0])+2):
		if i == 0 or i == len(environment)+1:
			row.append(create_state(0, False, True))
			continue
		if j == 0 or j == len(environment[0])+1:
			row.append(create_state(0, False, True))
			continue
		if environment[i-1][j-1] == '.':
			row.append(create_state(non_terminal_reward, False, False))
		elif environment[i-1][j-1] == 'X':
			row.append(create_state(0, False, True))
		else:
			row.append(create_state(float(environment[i-1][j-1]), True, False))
	states.append(row)

for k in range(K):
	for i in range(1, len(environment)+1):
		for j in range(1, len(environment[0])+1):
			if states[i][j]["isTerminalState"]:
				states[i][j]["utility"] = states[i][j]["reward"]
				continue
			if states[i][j]["isBlocked"]:
				continue
			results = []
			result = 0

			# check for top motion
			# top
			if states[i-1][j]["isBlocked"] == False:
				result += 0.8 * states[i-1][j]["utility"]
			else:
				result += 0.8 * states[i][j]["utility"]

			# left
			if states[i][j-1]["isBlocked"] == False:
				result += 0.1 * states[i][j-1]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			# right
			if states[i][j+1]["isBlocked"] == False:
				result += 0.1 * states[i][j+1]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			results.append(result)

			# check for bottom motion
			result = 0
			
			# bottom
			if states[i+1][j]["isBlocked"] == False:
				result += 0.8 * states[i+1][j]["utility"]
			else:
				result += 0.8 * states[i][j]["utility"]

			# left
			if states[i][j-1]["isBlocked"] == False:
				result += 0.1 * states[i][j-1]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			# right
			if states[i][j+1]["isBlocked"] == False:
				result += 0.1 * states[i][j+1]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			results.append(result)

			# check for left motion
			result = 0

			# left
			if states[i][j-1]["isBlocked"] == False:
				result += 0.8 * states[i][j-1]["utility"]
			else:
				result += 0.8 * states[i][j]["utility"]

			# top
			if states[i-1][j]["isBlocked"] == False:
				result += 0.1 * states[i-1][j]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			# bottom
			if states[i+1][j]["isBlocked"] == False:
				result += 0.1 * states[i+1][j]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			results.append(result)

			# check for right motion
			result = 0

			# right
			if states[i][j+1]["isBlocked"] == False:
				result += 0.8 * states[i][j+1]["utility"]
			else:
				result += 0.8 * states[i][j]["utility"]

			# top
			if states[i-1][j]["isBlocked"] == False:
				result += 0.1 * states[i-1][j]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			# bottom
			if states[i+1][j]["isBlocked"] == False:
				result += 0.1 * states[i+1][j]["utility"]
			else:
				result += 0.1 * states[i][j]["utility"]

			results.append(result)


			states[i][j]["utility"] = states[i][j]["reward"]+gamma* max(results)

for i in range(1, len(environment)+1):
	for j in range(1, len(environment[0])+1):
		print("%6.3f," % (states[i][j]["utility"]), end=" ")
	print()