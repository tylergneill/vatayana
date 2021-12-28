# Sample results of full-book Vātāyana search (NBhū, 598 pp.)

This simple html table is the output of Vātāyana's offline batch processing (see batch_processing.ipynb). All corpus texts were chosen as “priority”, and threshold values are N1 = 15% and N2 = 200.

[Click here to view the table in simple HTML](https://github.com/tylergneill/vatayana/blob/main/assets/nbhu_sample_results/similarity_results_entire_nbhu.html). The four scores shown are topic score relative to the entire corpus, topic score relative to only chosen “priority” texts, TF-IDF scores for those relatively few texts crossing the N1 threshold for topic scores, and Smith-Waterman alignment scores for those even fewer texts crossing the N2 threshold for TF-IDF scores.