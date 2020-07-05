import pandas as pd 
import matplotlib.pyplot as plt
import os

# QphH dataframe
folder = 'tpch_results/'
outputs = [file for file in os.walk(folder)][0][2]
averages = list()
for file in outputs:
    result = pd.read_table(folder+file, header=None, names=['num', 'power', 'throughput', 'qphh'], sep=',')
    # Pop num column
    result.pop('num')
    # Drop min and max
    result = result.drop(result['qphh'].idxmax())
    result = result.drop(result['qphh'].idxmin())
    averages.append((file[:-4], result['qphh'].mean()))
averages = pd.DataFrame(averages, columns=['Index configuration', 'QphH@1GB'])

# Plot QphH
averages.sort_values('QphH@1GB').plot(kind='bar', x='Index configuration', legend=False)
plt.ylim([950, 1100])
plt.ylabel('QphH@1GB')
plt.xticks(rotation=35)
# plt.show()
plt.tight_layout()
plt.savefig("qphh.pdf")
plt.clf()

# -----------------------------------------------------------------------------------

# Index sizes dataframe
sizes = pd.read_table('data/index_size.txt', sep=',')

# Plot index sizes
sizes.sort_values('Size').plot(kind='bar', x='Index configuration', legend=False)
plt.ylabel('Size in MB')
plt.xticks(rotation=35)
# plt.show()
plt.tight_layout()
plt.savefig("size.pdf")
plt.clf()

# -----------------------------------------------------------------------------------

# Ratios dataframe
ratios = pd.DataFrame(averages['Index configuration'])
ratios['Ratio'] = averages['QphH@1GB']/sizes['Size']

# Plot ratios
ratios.sort_values('Ratio').plot(kind='bar', x='Index configuration', legend=False)
plt.ylabel('Ratio')
plt.xticks(rotation=35)
# plt.show()
plt.tight_layout()
plt.savefig("ratio.pdf")
plt.clf()
