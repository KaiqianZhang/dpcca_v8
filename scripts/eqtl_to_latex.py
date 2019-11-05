import csv
from   decimal import Decimal

with open('eqtl_results.tsv') as f:
	reader = csv.reader(f, delimiter='\t')
	next(reader)
	for line in reader:
		tiss = line[0].replace('_', '\\_')
		snp  = line[2].replace('_', '\\_')
		comp = line[3].replace('row', '')
		# print(line)
		pval = '{:.2E}'.format(Decimal(line[6]))
		fdr  = '{:.2E}'.format(Decimal(line[7]))
		msg  = '%s & %s & %s & %s & %s \\\\' % (tiss, snp, comp, pval, fdr)
		print(msg)