import numpy as np

def read_file(content) -> dict: 
    with open(content, 'r') as file:
        data = file.readlines()
        data = [line.strip().split() for line in data]
        graph = {}
        for n1, n2 in data:
            if n1 not in graph:
                graph[n1] = []
            if n2 not in graph:
                graph[n2] = []
            graph[n1].append(n2)
            graph[n2].append(n1)
    return graph


def bfs(graph, s):
    stack = []
    predecessors = {v: [] for v in graph}
    sigma = {v: 0 for v in graph}
    distance = {s: 0}
    sigma[s] = 1
    queue = [s]
    while queue:
        v = queue.pop(0)
        stack.append(v)
        dv = distance[v]
        sigmav = sigma[v]
        for w in graph[v]:
            if w not in distance:
                queue.append(w)
                distance[w] = dv + 1
            if distance[w] == dv + 1:
                sigma[w] += sigmav
                predecessors[w].append(v)
    return stack, predecessors, sigma


def calculate_betweenness_centrality(graph):
    betweenness = {}
    for edge in graph.keys():
        betweenness[edge] = 0
    for s in graph:
        stack, predecessors, sigma = bfs(graph, s)
        betweenness = calculate_delta(graph, stack, predecessors, sigma, betweenness, s)
    return betweenness


def calculate_delta(graph, stack, predecessors, sigma, betweenness, s):
    delta = {v: 0 for v in graph}
    while stack:
        w = stack.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in predecessors[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def initialize_pagerank(graph):
    n = len(graph)
    nodes = sorted(list(graph.keys()))
    A = np.zeros((n, n))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if nodes[j] in graph[nodes[i]]:
                A[i][j] = 1
    dm = np.diag(A.sum(axis=1))
    PR = np.ones(n) / n
    return A, dm, PR, nodes


def pagerank_centrality(graph, alpha=0.85, beta=0.15, e=1e-6):
    A, dm, PR, nodes = initialize_pagerank(graph)
    n = len(PR)
    prev_PR = np.zeros(n)
    while np.sum(np.abs(prev_PR - PR)) > e:
        prev_PR = PR
        PR = alpha * A.T @ np.linalg.inv(dm) @ PR + beta * np.ones(n)
        PR = PR / np.sum(np.abs(PR))
    pagerank = {}
    for i in range(len(nodes)):
        pagerank[nodes[i]] = PR[i]
    return pagerank


def top_nodes(nodes):
    top_node = sorted(nodes.items(), key=lambda x: x[1], reverse=True)[:10]
    top_node = ", ".join([f"{node[0]}: {node[1]}" for node in top_node])
    return top_node


def main():
    graph = read_file('3. data.txt')
    betweenness = calculate_betweenness_centrality(graph)
    betweenness_nodes = top_nodes(betweenness)
    print("Top 10 nodes by betweenness centrality:")
    print(betweenness_nodes)
    pagerank = pagerank_centrality(graph)
    pagerank_nodes = top_nodes(pagerank)
    print("Top 10 nodes by PageRank centrality:")
    print(pagerank_nodes)


if __name__ == '__main__':
    main()