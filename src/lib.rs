#![allow(non_snake_case, non_camel_case_types, unused_mut, unused_imports , dead_code )]
/*
Mostly using vectors because, using arrays really makes every thing complicated and less flexible, 
we are just using the indexing feature and not continuouslly changing the size of the vector,
so we do not have any kind of considerable performance difference.
*/

pub mod mlalgos {
    
    pub mod k_cluster {
        use crate::n_dimen_algos::max_distance_between_sets;
        use crate::{n_dimen_algos::distance_between};
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
                
                print!("{}", closest_centroid_index);
                closest_centroid_index as u32
            }

            
        }
        //creating a struct, which stores all the info about the present k_mean.
        fn new_df(csv_file_path : &str ,K : usize, threshold : f32 ,lower_limit : f32 , upper_limit : f32) -> k_means_spec {
            let data = csv_to_df(csv_file_path).unwrap();
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
        pub fn k_means (csv_file_path : &str,
                        k_value : usize,
                        lower_limit : f32,
                        upper_limit : f32,
                        Threshold : f32, ) -> k_means_spec
        {

            //mut, because we will change the centroid values, after every iteration.
            let mut present_dataFrame = new_df(csv_file_path, k_value, Threshold, lower_limit, upper_limit);//now we have a data_set and its specifications to work on.
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

                //if we have reached the end of the iteration, print the iteration number.
                println!("{} iteration" , count);
                count += 1;
                
            }

            present_dataFrame

        }

        //This function will take the entire list of centroids and also the entire dataset , edits the individual sample points.
        //i.e sets the associative field to the nearest centroid.
        fn k_cluster(total_df : &mut k_means_spec ) -> () {
            //present_nearest:
            //we store the cluster point and it's distance , "if" after the next iteration 
            //we found out that this point is nearer to another centroid we update the present nearest.
            //iterating through every point to see for which cluster it belongs to.
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
        //while calculating the average they will , set , keep your fingers crossed for it atleast catches one sample point.maybe will update in next version.
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
        use csv::ReaderBuilder;

        //This function opens the csv and creates the data set, from the values in the csv,
        //output is a result so do not forget to unwrap.//we will not check if it is a csv file or not so be 
        //a little bit careful , when debugging.
        fn csv_to_df(file_path : &str) -> Result<Vec<sample_point> , Box<dyn Error>> {

            let file_system = File::open(file_path)?; 
            let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file_system);//we take like it always contains headers.
        
            //I know this code looks really unreadable but there is nothing more to work on.
            let full_dataset: Vec<sample_point> = reader.records()//records gives an iterator which gives out one row at a time, from the csv we provided.
                                                    .map(| record| -> Result<sample_point , Box<dyn Error>>{//first we take a row,
                                                    let mut this_point = sample_point { data : vec![],
                                                                                        associated_cluster : None };
                                                    let r = record?;
                                                    this_point.data = r.iter().map(|s| s.parse::<f32>()).collect::<Result<Vec<f32>, _ >>()?;//turning each value into an f32,cause the default will be in the form of a string.
                                                    Ok(this_point)//and turn it into a sample point, with its data as this record.
                                                    }).collect::<Result<Vec<sample_point>, _ >>()?;//we aollect all these sample points as a dataset.

        

        Ok(full_dataset)

        }




    }

}

pub mod n_dimen_algos {

    ///this function should take in two same sized vectors and give out the max distance between any two respective positions.
    pub fn max_distance_between_sets ( previous_centroid : &Vec<Vec<f32>> , current_centroid : &Vec<Vec<f32>>) -> f32 {

        assert!(previous_centroid.len() == current_centroid.len() , "The input vectors do not contain the same number of points!!!!");

        let mut out_vec = vec![];

        for i in 0..previous_centroid.len() {
            out_vec.push(distance_between(previous_centroid[i].as_ref() , current_centroid[i].as_ref()));
        }
        let mut max = 0_f32;

        for ty in out_vec {
            if ty > max {
                max = ty
            }
        }

        max

    }

    ///takes two arrays of length n(n-dimensional points), returns the euclidean distance between them.
    pub fn distance_between( point_1 : &Vec<f32> , point_2 : &Vec<f32> ) -> f32 {
        
        assert!(point_1.len() >= 1 && point_2.len() >= 1 , "You should have atleast 1 feature for the sample point");
        assert!(point_1.len() == point_2.len() , "Both the points should have the same number of features");

        let dimensions = point_1.len();
        let mut sum_of_squares_of_difference = 0.0;

        for i in 0..dimensions {
            sum_of_squares_of_difference += (point_1[i] - point_2[i]).powf(2.0);
        }

        sum_of_squares_of_difference.sqrt()

    }
    

}

#[cfg(test)]
mod test {
    use crate::{n_dimen_algos::{distance_between}};

    #[test]
    fn distance() {
        
        let p1 = vec![0.0 , 0.0 , 0.0 , 4.0];
        let p2 = vec![1.0 , 1.0 , 1.0 , -9.0];

        dbg!(distance_between(&p1, &p2));

    }//working

    #[test]
    fn csv_test_print() {

        //code deleted due to private functions.

    }//working

    #[test]
    fn df_thing() {
        //let hava = new_df("C:/Users/HARSHA/Downloads/wine-clustering.csv", 2 ,0.01 , 1.0, 10.0);

        //print!("{:?}", hava);
    }//working, do not test again the code has gone private.

}