
# load the dataset
dataset = []
with open('data/cat-facts.txt', 'r', encoding='utf-8') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')