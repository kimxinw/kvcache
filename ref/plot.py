import csv, matplotlib.pyplot as plt
x, c, r = [], [], []
for row in csv.DictReader(open('data/bench.csv')):
    x.append(int(row['seq_len'])); 
    c.append(float(row[' cache_us'])); 
    r.append(float(row[' recompute_us']))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(x, c, 'o-', label='KV cache  O(t)'); a1.plot(x, r, 's-', label='recompute  O(t²)')
a1.set_xlabel('sequence length'); 
a1.set_ylabel('per-step latency (µs)'); 
a1.legend(); 
a1.set_title('decode cost')

a2.plot(x, [ri/ci for ri, ci in zip(r, c)], 'd-', color='C2')
a2.set_xlabel('sequence length'); 
a2.set_ylabel('speedup ×'); 
a2.set_title('KV cache speedup')

plt.tight_layout(); plt.savefig('data/bench.png', dpi=130); 
print('saved data/bench.png')