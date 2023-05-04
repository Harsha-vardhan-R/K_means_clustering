#![allow(non_snake_case, unused_imports, unused_variables, non_upper_case_globals, dead_code, unused_mut)]
use K_means_clustering::mlalgos::k_cluster::generate_k_centroids;
use rand::prelude::*;


//Threshold for the clustering resolution.
const THRESHOLD : f32 = 0.01;
//the number of clusters we want to create.
const K : usize = 2;
//below both depend on the dataset.
const  number_of_features : usize = 2;
const  number_of_samples : usize = 10;
//The below is for calculating random centroids in the range of the the dataset, 
//to do this , normally observe the data set and choose a really leas value and a really big value from it.


fn main() {
    
    /*
    this array should be of the same size as the number of the sample points,
    and respectively should represent which point is related to which cluster.
    here they are k different clusters.
    */
    let mut associated_cluster : [usize ; number_of_samples] = Default::default();

    //given dataset to train on.
    let dataset : [[f32 ; number_of_features] ; number_of_samples] = Default::default();
    let mut k_cluster_centroid : [[f32 ; number_of_features] ; K] = Default::default();


    //firstly we calculate the the centroids randomly,the lower_limit and upper_limit should be manually selected, by observing the data set a little bit.
    let centroids = generate_k_centroids(K, number_of_features, 0.0, 20.0);

    //Now we loop calling the 
    


}
