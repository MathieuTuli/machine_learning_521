lines = []
with open('assignment3_part1.3.2.py', 'r') as f:
    for line in f:
        line = line.replace('\t', '    ')
        lines.append(line)

with open('assignment3_part1.3.2.py', 'w') as f:
    f.writelines(lines)
