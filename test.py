from decoding import DraftTree, Node
import torch

device = "cpu"

# Create
dtree = DraftTree()

n = [Node(i) for i in range(7)]

dtree.root.add_child(n[0])
n[0].add_child(n[1])

n[1].add_child(n[2])
n[1].add_child(n[3])

dtree.root.add_child(n[4])
n[4].add_child(n[5])
n[5].add_child(n[6])

dtree.done()

# Print
dtree.debug()

# Methods
print("tokens =", dtree.tokens(device))
print("position_ids =", dtree.position_ids(10, device))

mask = torch.full(
    size=(dtree.size(), dtree.size()),
    fill_value=-10.0,
    dtype=torch.bfloat16,
    device=device,
)
dtree.zero_mask(mask)
print("mask =")
print(mask)
