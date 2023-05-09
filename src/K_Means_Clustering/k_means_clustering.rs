
#[derive(Debug)]
pub struct sample_point {
    pub data : Vec<f32>,
    pub associated_cluster : Option<u32>,
}

#[derive(Debug)]
pub struct k_means_spec<'a> {
    csv_file_path : &'a str,
    data : Vec<sample_point>,
    centroids : Vec<Vec<f32>>,
    k : usize,
    number_of_features : usize,
    number_of_samples : usize,
    pub threshold : f32,
    pub visuals : bool,
}

use core::f32;

impl k_means_spec<'_> {
    
    pub fn print(&self) {
        println!("{:?}", self );
    }

    pub fn encoding_names(&mut self) {

    }
    pub fn print_associates(&self) {
        for associates in &self.data {
            print!("{}," , associates.associated_cluster.unwrap());
        }
    }
    //working , thank god!!!!!
    pub fn update_centroids(&mut self) {
        let mut centroids_with_count: Vec<(Vec<f32>, usize)> = vec![(vec![0.0; self.number_of_features], 0); self.k];
        
        // Sum the data points in each cluster
        for point in &self.data {
            if let Some(cluster_index) = point.associated_cluster {
                let centroid = &mut centroids_with_count[cluster_index as usize].0;
                for (i, feature) in point.data.iter().enumerate() {
                    centroid[i] += feature;
                }
                centroids_with_count[cluster_index as usize].1 += 1;
            }
        }
        
        // Compute the average of the data points in each cluster to get the new centroids
        //be very careful, if a cluster does not contain any point then dividing the sum by zero will give out NaN.
        //now applying , if a centroid is not associated with any kind of point , we do not change it's value.
        let mut new_centroids : Vec<Vec<f32>> = vec![];

        for (i , avgs_with_counts) in centroids_with_count.iter().enumerate() {
            //Normally this should not happen if every point is perfectly random in all the features, but that is not the case here , so......
            match avgs_with_counts {
                ( _ , 0) => new_centroids.push(self.centroids[i].clone()),//if the count is zero , we do not modify the centroids.
                _ => new_centroids.push(avgs_with_counts.0.iter().map(|each| each / (avgs_with_counts.1 as f32)).collect()), //else we will do the average thing
            }

        }

        self.centroids = new_centroids
        

    }
    
    ///this is for predicting.
    pub fn predict(&self, x: &Vec<f32>) -> u32 {
        let mut present_min_dist_with = f32::INFINITY;
        let mut closest_centroid_index = 0;
        
        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = distance_between(x, centroid);
            if dist < present_min_dist_with {
                present_min_dist_with = dist;
                closest_centroid_index = i;
            }
        }
        
        println!("{}", closest_centroid_index);
        closest_centroid_index as u32
    }

    
}
//creating a struct, which stores all the info about the present k_mean.
fn new_df(csv_file_path : & str ,K : usize, threshold : f32 ,lower_limit : f32 , upper_limit : f32 , which_features: Vec<usize>) -> k_means_spec {
    let data = csv_to_df(csv_file_path , which_features).unwrap();
    //we are calculating the number of features after making the data frame, so we need not change the size while generating the centroids.
    let number_of_features = data[0].data.len();
    let number_of_samples = data.len();
    //creating and returning a new k_means_spec struct.
    k_means_spec { csv_file_path: csv_file_path,
                   data: data,
                   centroids : generate_k_centroids(K, number_of_features, lower_limit, upper_limit),
                   k: K,
                   number_of_features: number_of_features,
                   number_of_samples: number_of_samples,
                   threshold : threshold,
                   visuals: true
    }
}

//This is the main logic behind, users should use this.
//Lower_limit and upper limit will be used in the random generation function.
pub fn k_means(csv_file_path : &str,
                k_value : usize,
                lower_limit : f32,
                upper_limit : f32,
                Threshold : f32,
                which_features: Vec<usize> ) -> k_means_spec
{

    //mut, because we will change the centroid values, after every iteration.
    let mut present_dataFrame = new_df(csv_file_path, k_value, Threshold, lower_limit, upper_limit , which_features);//now we have a data_set and its specifications to work on.
    let mut count = 1;
    //clustering in k means until we get the centroid points moving less than threshold value after one iteration.
    //main loop
    loop {
        //saving the points , to compare them afterwards.
        let previous_centroids = present_dataFrame.centroids.clone();
        println!("{:?}", present_dataFrame.centroids);
        
        k_cluster(&mut present_dataFrame);
        //changing the centroids based on the present sample points association, this is a method call directly implemented on the K_means_spec.
        present_dataFrame.update_centroids();

        //if the largest change between any centroid respective to its previous position is leaa than the threshold value, we will break out of the loop.
        if max_distance_between_sets(&previous_centroids , &present_dataFrame.centroids) < present_dataFrame.threshold {
            println!("Done!");
            break;
        };
        present_dataFrame.print_associates();                
        print!("\n");
        //if we have reached the end of the iteration, print the iteration number.
        println!("{} iteration done" , count);
        count += 1;
        
    }

    present_dataFrame

}

//This function will take the entire list of centroids and also the entire dataset.
//i.e sets the associative field to the nearest centroid.
fn k_cluster(total_df : &mut k_means_spec ) -> () {
     
    for sample_point in 0..total_df.number_of_samples {
        let mut present_nearest : (u32 , f32) = (1000 , std::f32::INFINITY);//initialising with obscure values so that this will for sure be updated.
        //now we have one sample in our hand, time to find out the nearest centroid to this.
        for cluster in 0..total_df.k {
            //Now we have a centroid and a sample point ;), time to to find out the distance.
            let dist_now = distance_between(&total_df.data[sample_point].data, &total_df.centroids[cluster]);
            
            if dist_now < present_nearest.1 {
                present_nearest = (cluster as u32 , dist_now);
            }

        }
        //storing the nearest centroid in the associated cluster field.
        total_df.data[sample_point].associated_cluster = Some(present_nearest.0);

    }
    

} 

use fastrand::Rng;
//This only generates random numbers , but do not worry about different ranges in different features, even if it contains one sample point,
//while calculating the average they will set , keep your fingers crossed for it atleast catches one sample point.maybe will update in next version.
fn generate_k_centroids(number_of_clusters : usize ,
                        number_of_features : usize ,
                        lower_limit : f32 , 
                        upper_limit : f32) -> Vec<Vec<f32>> 
{
    
    //creating an empty array cause now we know the size of the output.
    let mut out_centroids:Vec<Vec<f32>> = Vec::new();
    
    for _centroids in 0..number_of_clusters {
        
        //create this point and push into the all centroids list.
        let mut this_cluster:Vec<f32> = Vec::new();
        for _centroid_feature in 0..number_of_features {
            let mut rng = Rng::new();
            let random_f32 = rng.f32();
            this_cluster.push(lower_limit + (random_f32 * (upper_limit - lower_limit)));
        }
        out_centroids.push(this_cluster);
    }
    dbg!(&out_centroids);
    out_centroids

}

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use csv::ReaderBuilder;

use crate::n_dimen::distance_between;
use crate::n_dimen::max_distance_between_sets;

pub fn csv_to_df(
    file_path: &str,
    which_features: Vec<usize> ) -> Result<Vec<sample_point>, Box<dyn Error>> {

    let file_system = File::open(file_path)?;
    let reader = BufReader::new(file_system);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let full_dataset: Vec<sample_point> = csv_reader
        .records()
        .filter_map(|record| {
            let mut this_point = sample_point {
                data: vec![],
                associated_cluster: None,
            };
            let record = record.ok()?;
            if which_features.is_empty() {
                // If which_features is empty, consider all the columns
                this_point.data = record
                    .iter()
                    .map(|s| s.parse::<f32>())
                    .collect::<Result<Vec<f32>, _>>()
                    .ok()?;
            } else {
                // Consider only the columns specified by which_features
                this_point.data = which_features
                    .iter()
                    .map(|&i| record.get(i))
                    .collect::<Option<Vec<&str>>>()?
                    .iter()
                    .map(|&s| s.parse::<f32>())
                    .collect::<Result<Vec<f32>, _>>()
                    .ok()?;
            }
            Some(this_point)
        })
        .collect::<Vec<_>>();

    Ok(full_dataset)
}