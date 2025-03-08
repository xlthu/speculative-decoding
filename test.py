from decoding import DraftTree
import torch

device = "cpu"

# Create
dtree = DraftTree()

n = [dtree.new_node(i) for i in range(7)]

dtree.root.add(n[0])
n[0].add(n[1])

n[1].add(n[2])
n[1].add(n[3])

dtree.root.add(n[4])
n[4].add(n[5])
n[5].add(n[6])

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
