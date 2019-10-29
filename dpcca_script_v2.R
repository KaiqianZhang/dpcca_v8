# for running on della
# test files


#uses brians' set up
#example for Adipose_Subcutaneous
#todo make parameter

.libPaths("/home/bj5/R/x86_64-redhat-linux-gnu-library/3.1")
genotype_folder = '/tigress/BEE/RNAseq/RNAseq_dev/Data/Genotype/gtex_v8/dosage/'
Adipose_Subcutaneous = 'v8_WholeGenomeSeq_838Indiv_Analysis.Adipose_Subcutaneous.final.tsv'
genotype_file_location = paste(genotype_folder, Adipose_Subcutaneous, sep="")


expression_folder = '/tigress/BEE/usr/biancad/DCCA/data/'
expression_name = 'expression_matrix_Adipose_Subcutaneous_latent_dim-10.txt'
expression_file_location = paste(expression_folder, expression_name, sep="")

output_folder = '/tigress/BEE/usr/biancad/DCCA/output/'
output_name   = 'Adipose_Subcutaneous.txt'
output_file   = paste(output_folder, output_name, sep="") 

if (FALSE) {

#warning, the two files dont have matching ID
# it takes too long to align the files by doing the following
#load the SNP data
#genotype_matrix = read.table(genotype_file_location, header = TRUE, sep = '\t', stringsAsFactors = FALSE)


# Make sure genotype and expression matrix columns line up
# takes a bit to load
#expression_matrix = expression_matrix[,(colnames(expression_matrix) %in% colnames(genotype_matrix))]
#genotype_matrix = genotype_matrix[,sapply(colnames(expression_matrix), function(x) {match(x, colnames(genotype_matrix))})] 


#if things were aligned to this:

# CREATE THE SNP file
library(MatrixEQTL)

snps = SlicedData$new()
SNP_file_name = genotype_file_location
snps$fileDelimiter = "\t";
snps$fileOmitCharacters = "NA";
snps$fileSkipRows = 1;
snps$fileSkipColumns = 1;       # one column of row labels; MAYBE WE CAN CHANGE THIS to align things properly
snps$fileSliceSize = 2000;      # read file in pieces of 2,000 rows
snps$LoadFile(SNP_file_name);  #takes a very long time ~ 15 min


# CREATE THE GENEEXPRESSION file
expression_matrix = read.csv(file = expression_file_location, header = TRUE, sep = '\t', stringsAsFactors = FALSE)
rownames(expression_matrix) = expression_matrix$gene_id
expression_matrix = expression_matrix[,2:dim(expression_matrix)[2]]
gene = SlicedData$new()
gene$CreateFromMatrix(as.matrix(expression_matrix));


# OTHER PARAMS
pvOutputThreshold = 1e-5
useModel = modelLINEAR;

me = Matrix_eQTL_engine(snps = snps, 
+                    gene = gene, 
+                    cvrt = SlicedData$new(), 
+                    output_file_name = output_file, 
+                    pvOutputThreshold = pvOutputThreshold, 
+                    useModel = useModel, 
+                    errorCovariance = numeric(), 
+                    verbose = TRUE,
+                    pvalue.hist = 100,
+                    min.pv.by.genesnp = FALSE,
+                    noFDRsaveMemory = FALSE)



 me_all = me$all
}
 # has to be ran for permuted data as well! for FDR calibration
 # head(me_all$eqtls)
 # this one contains the eQTLs and the factors, the effect size and the pvalue
 # pvalue can be more stringent
 # 16629  rs11200189    7 -3.111938 2.066000e-03 1.0000000 -1.944349e-02
#16630   rs2250163    6 -3.111903 2.066236e-03 1.0000000 -2.662733e-02
#16631  rs11245270   11 -3.111902 2.066247e-03 1.0000000 -3.110772e-02
#16632  rs12357014    7  3.111900 2.066261e-03 1.0000000  1.096803e-02
#16633 rs184918566   19 -3.111898 2.066275e-03 1.0000000 -2.355032e-02
#16634  rs72811689   23 -3.111889 2.066335e-03 1.0000000 -2.557929e-04
#16635  rs11196731   24  3.111880 2.066397e-03 1.0000000  2.067620e-04
#16636  rs11101852   16  3.111870 2.066461e-03 1.0000000  2.577901e-02


 


