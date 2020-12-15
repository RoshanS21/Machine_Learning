# Roshan Shrestha


environment_file = "environment2.txt"
non_terminal_reward = -0.04
gamma = 1
K = 20

environment = []

def create_state(terminalReward, terminalState, blocked):
	state = {"reward": terminalReward, "utility": 0,\
		"isTerminalState": terminalState, "isBlocked": blocked}
	return state

import csv
row_count = 0
column_count = 0

with open(environment_file, newline='') as csvfile:
	line = csv.reader(csvfile, delimiter=',', quotechar='|')
	environment_row = []
	for row in line:
		column_count = 0
		environment_row = []
		for element in row:
			if element == '.':
				state = create_state(non_terminal_reward, False, False)
				environment_row.append(state)
			elif element == 'X':
				state = create_state(0, False, True)
				environment_row.append(state)
			else:
				state = create_state(float(element), True, False)
				environment_row.append(state)
			column_count += 1
		environment.append(environment_row)
		row_count += 1

for k in range(K):
	for i in range(row_count):
		for j in range(column_count):
			if environment[i][j]["isTerminalState"] == True:
				environment[i][j]["utility"] = environment[i][j]["reward"]
				continue
			if environment[i][j]["isBlocked"] == True:
				continue
			surround_states = []
			# up
			s = 0
			# same state
			if i-1 < 0:
				s += 0.8*environment[i][j]["utility"]
			if i-1 >= 0 and environment[i-1][j]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			if j-1 < 0:
				s += 0.1*environment[i][j]["utility"]
			if j-1 >=0 and environment[i][j-1]["isBlocked"] == True:
				s += 0.1*environment[i][j]["utility"]
			if j+1 >= column_count:
				s += 0.1*environment[i][j]["utility"]
			if j+1 < column_count and environment[i][j+1]["isBlocked"] == True:
				s += 0.1*environment[i][j]["utility"]
			# above state
			if i-1 >= 0 and environment[i-1][j]["isBlocked"] == False:
				s += 0.8*environment[i-1][j]["utility"]
			# left state
			if j-1 >= 0 and environment[i][j-1]["isBlocked"] == False:
				s += 0.1*environment[i][j-1]["utility"]
			# right state
			if j+1 < column_count and environment[i][j+1]["isBlocked"] == False:
				s += 0.1*environment[i][j+1]["utility"]
			surround_states.append(s)

			# down
			s = 0
			# same state
			if i+1 >= row_count:
				s += 0.8*environment[i][j]["utility"]
			if i+1 < row_count and environment[i+1][j]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			if j-1 < 0:
				s += 0.1*environment[i][j]["utility"]
			if j-1 >=0 and environment[i][j-1]["isBlocked"] == True:
				s += 0.1*environment[i][j]["utility"]
			if j+1 >= column_count:
				s += 0.1*environment[i][j]["utility"]
			if j+1 < column_count and environment[i][j+1]["isBlocked"] == True:
				s += 0.1*environment[i][j]["utility"]
			# below state
			if i+1 < row_count and environment[i+1][j]["isBlocked"] == False:
				s += 0.8*environment[i+1][j]["utility"]
			# left state
			if j-1 >= 0 and environment[i][j-1]["isBlocked"] == False:
				s += 0.1*environment[i][j-1]["utility"]
			# right state
			if j+1 < column_count and environment[i][j+1]["isBlocked"] == False:
				s += 0.1*environment[i][j+1]["utility"]
			surround_states.append(s)

			# left
			s = 0
			# same state
			if j-1 < 0:
				s += 0.8*environment[i][j]["utility"]
			if j-1 >= 0 and environment[i][j-1]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			if i-1 < 0:
				s += 0.1*environment[i][j]["utility"]
			if i-1 >=0 and environment[i-1][j]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			if i+1 >= row_count:
				s += 0.1*environment[i][j]["utility"]
			if i+1 < row_count and environment[i+1][j]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			# left state
			if j-1 >= 0 and environment[i][j-1]["isBlocked"] == False:
				s += 0.8*environment[i][j-1]["utility"]
			# above state
			if i-1 >= 0 and environment[i-1][j]["isBlocked"] == False:
				s += 0.1*environment[i-1][j]["utility"]
			# below state
			if i+1 < row_count and environment[i+1][j]["isBlocked"] == False:
				s += 0.1*environment[i+1][j]["utility"]
			surround_states.append(s)

			# right
			s = 0
			# same state
			if j+1 >= column_count:
				s += 0.8*environment[i][j]["utility"]
			if j+1 < column_count and environment[i][j+1]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			if i-1 < 0:
				s += 0.1*environment[i][j]["utility"]
			if i-1 >=0 and environment[i-1][j]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			if i+1 >= row_count:
				s += 0.1*environment[i][j]["utility"]
			if i+1 < row_count and environment[i+1][j]["isBlocked"] == True:
				s += 0.8*environment[i][j]["utility"]
			# left state
			if j+1 < column_count and environment[i][j+1]["isBlocked"] == False:
				s += 0.8*environment[i][j+1]["utility"]
			# above state
			if i-1 >= 0 and environment[i-1][j]["isBlocked"] == False:
				s += 0.1*environment[i-1][j]["utility"]
			# below state
			if i+1 < row_count and environment[i+1][j]["isBlocked"] == False:
				s += 0.1*environment[i+1][j]["utility"]
			surround_states.append(s)

			environment[i][j]["utility"] = environment[i][j]["reward"] + gamma\
			* max(surround_states)

for environment_row in environment:
	for element in environment_row:
		print("%6.3f" % (element["utility"]), end=" ")
	print()
