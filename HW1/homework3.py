import numpy as np
import networkx as nx
import heapq
import time

#Breadth-first-search
def bfs(matrixtran, start_state, end_states, info, outputFile):
	queue = list()
	start_node = list(map(int, start_state)) 
	queue.append(start_node)
	count = 0
	file = open(outputFile, 'w')

	for i in end_states:
		resstring = ""
		flag = 0
		count = count + 1
		settling_site = [int(k) for k in i] 
		q = []
		visited = [[False for i in range(rows)] for j in range(columns)]
		dist = [[1000000000 for i in range(rows)] for j in range(columns)]
		pred = [[-1 for i in range(rows)] for j in range(columns)]
		visited[start_node[0]][start_node[1]] = True
		dist[start_node[0]][start_node[1]] = 0
		q.append(start_node)

		while(len(q) != 0):
			u = q[0]
			q.pop(0)
			neighbors = lambda x, y : [[x2, y2] for x2 in range(x-1, x+2)
				for y2 in range(y-1, y+2)
				if (-1 < x <= columns and
					-1 < y <= rows and
					(x != x2 or y != y2) and
					(0 <= x2 <= columns) and
					(0 <= y2 <= rows))]
			arr = neighbors(u[0],u[1])

			for j in arr:
				flg = 0
				if j[0] < columns and j[1] < rows:
					if int(matrixtran[j[0]][j[1]]) <= 0:
						if int(matrixtran[u[0]][u[1]]) <=0:
							if visited[j[0]][j[1]] == False and abs(abs(int(matrixtran[u[0]][u[1]])) - abs(int(matrixtran[j[0]][j[1]]))) <= int(info[3]):
								visited[j[0]][j[1]] = True
								dist[j[0]][j[1]] = dist[u[0]][u[1]] + 1
								pred[j[0]][j[1]] = u
								q.append(j)
								flg = 1
								
						elif int(matrixtran[u[0]][u[1]]) > 0:
							if visited[j[0]][j[1]] == False and abs(0 - abs(int(matrixtran[j[0]][j[1]]))) <= int(info[3]):
								visited[j[0]][j[1]] = True
								dist[j[0]][j[1]] = dist[u[0]][u[1]] + 1
								pred[j[0]][j[1]] = u
								q.append(j)
								flg = 1
								
					elif int(matrixtran[j[0]][j[1]]) > 0:
						if int(matrixtran[u[0]][u[1]]) <= 0:
							if visited[j[0]][j[1]] == False and abs(abs(int(matrixtran[u[0]][u[1]])) - 0) <= int(info[3]):
								visited[j[0]][j[1]] = True
								dist[j[0]][j[1]] = dist[u[0]][u[1]] + 1
								pred[j[0]][j[1]] = u
								q.append(j)
								flg = 1
								
						else:
							if visited[j[0]][j[1]] == False:
								visited[j[0]][j[1]] = True
								dist[j[0]][j[1]] = dist[u[0]][u[1]] + 1
								pred[j[0]][j[1]] = u
								q.append(j)
								flg = 1			
				if flg == 1:
					if(j == settling_site):
									flag = 1
									path = []
									crawl = settling_site
									path.append(crawl)

									while(pred[crawl[0]][crawl[1]] != -1):
										path.append(pred[crawl[0]][crawl[1]])
										crawl = pred[crawl[0]][crawl[1]]

									for k in range(len(path)-1,-1,-1):
										for l in path[k]:
											resstring = resstring+str(l)+","
										resstring = resstring.strip(',')+' '
									if count == int(no_of_settling_sites):
										file.write(resstring.strip().strip(','))
									else:
										file.write(resstring.strip().strip(',') + '\n')
			

		if flag == 0:
			if count == int(no_of_settling_sites):
				file.write("FAIL")
			else:
				file.write("FAIL" + '\n')



#Uniform cost search
def ucs(matrix, matrixtran, start_state, end_states, info, outputFile):
	count = 0
	file = open(outputFile, 'w')
	G = nx.Graph()
	# Use a dictionary to keep track of the elements inside the frontier (queue)
	# explored = set()
	# print(start_node)
	for i in range(0,rows):
		for j in range(0,columns-1):
			G.add_edge((i,j),(i,j+1), weight = 10)
			G.add_edge((j,i),(j+1,i), weight = 10)

	for i in end_states:
		start_node = list(map(int, start_state))
		node = (0, start_node, [start_node])
		frontier = []
		frontierIndex = {}
		frontierIndex[tuple(node[1])] = [node[0], node[2]]
		heapq.heappush(frontier, node)
		explored = set()
		resstring = ""
		flag = 0
		count = count + 1
		settling_site = [int(k) for k in i]
		
		while frontier:
			if len(frontier) == 0:
				if count == int(no_of_settling_sites):
					file.write("FAIL")
				else:
					file.write("FAIL" + '\n')
			node = heapq.heappop(frontier)
			# Delete from the dicitonary the element that has beeen popped
			del frontierIndex[tuple(node[1])]
			# Check if the solution has been found
			# print(node[1])
			if node[1] == settling_site:
				flag = 1
				# print(flag)
				for j in node[2]:
					for k in j:
						resstring = resstring+str(k)+","
					resstring = resstring.strip(',')+' '
				if count == int(no_of_settling_sites):
					file.write(resstring.strip().strip(','))
				else:
					file.write(resstring.strip().strip(',') + '\n')

			explored.add(tuple(node[1]))
			# Get a list of all the child nodes of node
			neighbors = lambda x, y : [[x2, y2] for x2 in range(x-1, x+2)
					for y2 in range(y-1, y+2)
					if (-1 < x <= columns and
						-1 < y <= rows and
						(x != x2 or y != y2) and
						(0 <= x2 <= columns) and
						(0 <= y2 <= rows))]
			arr = neighbors(node[1][0],node[1][1])
			path = node[2]

			for child in arr:
				flg = 0
				if child[0] < columns and child[1] < rows:
			
					if int(matrixtran[child[0]][child[1]]) <=0:
						if int(matrixtran[node[1][0]][node[1][1]]) <=0:
							if abs(abs(int(matrixtran[node[1][0]][node[1][1]])) - abs(int(matrixtran[child[0]][child[1]]))) <= int(info[3]):
								path.append(child)
								flg = 1
								
						elif int(matrixtran[node[1][0]][node[1][1]]) > 0:
							if abs(0 - abs(int(matrixtran[child[0]][child[1]]))) <= int(info[3]):
								path.append(child)
								flg = 1
								
					elif int(matrixtran[child[0]][child[1]]) >0:
						if int(matrixtran[node[1][0]][node[1][1]]) <=0:
							if abs(abs(int(matrixtran[node[1][0]][node[1][1]])) - 0) <= int(info[3]):
								path.append(child)
								flg = 1
								
						else:
							path.append(child)
							flg = 1

				if flg == 1:
					# create the child node that will be inserted in frontier
					weight = G.get_edge_data(tuple(node[1]),tuple(child))
					if weight == None:
						weight = 14
					else:
						weight = G.get_edge_data(tuple(node[1]),tuple(child))["weight"]
					childNode = (node[0] + weight, child, path)
					# Check the child node is not explored and not in frontier thorugh the dictionary
					if tuple(child) not in explored and tuple(child) not in frontierIndex:
						heapq.heappush(frontier, childNode)
						frontierIndex[tuple(child)] = [childNode[0], childNode[2]]

					elif tuple(child) in frontierIndex: 
						# Checks if the child node has a lower path cost than the node already in frontier
						if childNode[0] < frontierIndex[tuple(child)][0]:
							nodeToRemove = (frontierIndex[tuple(child)][0], child, frontierIndex[tuple(child)][1])
							frontier.remove(nodeToRemove)
							heapq.heapify(frontier)
							del frontierIndex[tuple(child)]

							heapq.heappush(frontier, childNode)
							frontierIndex[tuple(child)] = [childNode[0], childNode[2]]
					path = path[:-1]


		if flag == 0:
			# print("here")
			if count == int(no_of_settling_sites):
				file.write("FAIL")
			else:
				file.write("FAIL" + '\n')



#A star Search
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.position == other.position

def astar(matrix, maze, start_state, end_states, info, outputFile):
	start = tuple(map(int, start_state))
	G = nx.Graph()
	count = 0
	start_node = Node(None, tuple(start))
	start_node.g = start_node.h = start_node.f = 0
	file = open(outputFile, 'w')
	
	for i in range(0,rows):
		for j in range(0,columns-1):
			G.add_edge((i,j),(i,j+1), weight = 10)
			G.add_edge((j,i),(j+1,i), weight = 10)

	for i in end_states:
		# print(i)
		resstring = ""
		count = count + 1
		flag = 0
		settling_site = tuple(int(k) for k in i)
		# print(settling_site)
		end_node = Node(None, tuple(settling_site))
		end_node.g = end_node.h = end_node.f = 0
		open_list = []
		closed_list = []
		open_list.append(start_node)
		outer_iterations = 0
		max_iterations = (len(maze) // 2) * 100

		while len(open_list) > 0:
			outer_iterations +=1
			current_node = open_list[0]
			current_index = 0
			for index, item in enumerate(open_list):
				if item.f < current_node.f:
					current_node = item
					current_index = index

			if outer_iterations > max_iterations:
				flag = 0
				break

			open_list.pop(current_index)
			closed_list.append(current_node)

			# print(len(closed_list))
			# print()
			if current_node == end_node:
				# print("here")
				flag = 1
				path = []
				current = current_node
				while current is not None:
					path.append(current.position)
					current = current.parent
				# path = path[::-1]
				for j in path[::-1]:
					for k in j:
						resstring = resstring+str(k)+","
					resstring = resstring.strip(',')+' '
				if count == int(no_of_settling_sites):
					file.write(resstring.strip().strip(','))
				else:
					file.write(resstring.strip().strip(',') + '\n')
				break

			children = []
			for new_position in [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]:
				node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
				if node_position[0] > (len(maze)-1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
				# if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) - 1) or node_position[1] < 0:
					continue

				new_node = Node(current_node, node_position)
				children.append(new_node)

			for child in children:
				if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
					continue

				weight = G.get_edge_data(current_node.position,child.position)
				if weight == None:
					weight = 14
				else:
					weight = G.get_edge_data(current_node.position,child.position)["weight"]

				if(int(maze[current_node.position[0]][current_node.position[1]]) > 0):
					if(int(maze[child.position[0]][child.position[1]]) <= 0):
						if abs(0 - abs(int(maze[child.position[0]][child.position[1]]))) <= int(info[3]):
							child.g = current_node.g + weight + abs(0 - abs(int(maze[child.position[0]][child.position[1]])))
						else:
							continue
					elif(int(maze[child.position[0]][child.position[1]]) > 0):

						child.g = current_node.g + weight + int(maze[child.position[0]][child.position[1]])

				elif(int(maze[current_node.position[0]][current_node.position[1]]) <= 0):
					if(int(maze[child.position[0]][child.position[1]]) > 0):
							
						child.g = current_node.g + weight + int(maze[child.position[0]][child.position[1]]) + abs(0 - abs(int(maze[current_node.position[0]][current_node.position[1]])))
					elif(int(maze[child.position[0]][child.position[1]]) <= 0):

						if abs(abs(int(maze[current_node.position[0]][current_node.position[1]])) - abs(int(maze[child.position[0]][child.position[1]]))) <= int(info[3]):
							child.g = current_node.g + weight + abs(abs(int(maze[current_node.position[0]][current_node.position[1]])) - abs(int(maze[child.position[0]][child.position[1]])))
						else:
							continue
			
				child.h = max(abs(child.position[0] - end_node.position[0]), abs(child.position[1] - end_node.position[1]))

				
				child.f = child.g + child.h
				# for open_node in open_list:
				# 	if child == open_node and child.g > open_node.g:
				# 	# if child == open_node and child.f > open_node.f:
				# 		continue

				if len([i for i in open_list if child == i and child.g > i.g]) > 0:
					continue

				open_list.append(child)
				# print(len(open_list))

		if flag == 0:
			if count == int(no_of_settling_sites):
				file.write("FAIL")
			else:
				file.write("FAIL" + '\n')




#Main program
inputFile = 'input.txt'
outputFile = 'output.txt'

info = list()
matrix = []
matrixtran = []
end_states = []

with open(inputFile) as f:
	for line in f.readlines():
		info.append(line.strip())

columns = int(info[1].split()[0])
rows = int(info[1].split()[1])

algorithm = info[0]
start_state = info[2].split()
no_of_settling_sites = info[4]

for i in range(5+int(no_of_settling_sites),5+int(no_of_settling_sites)+rows): 
	matrix.append(info[i].split())

# print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in matrix]))
matrixtran = np.transpose(matrix)

for i in range(5,5+int(no_of_settling_sites)):
	end_states.append(info[i].split())

if info[0]=='BFS':
	bfs(matrixtran, start_state, end_states, info, outputFile)
elif info[0]=='UCS':
	ucs(matrix, matrixtran, start_state, end_states, info, outputFile)
elif info[0]=='A*':
	astar(matrix, matrixtran, start_state, end_states, info, outputFile)

