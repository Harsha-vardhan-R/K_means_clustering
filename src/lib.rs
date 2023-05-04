#![allow(non_snake_case,unused_variables)]

pub mod mlalgos {
    
    pub mod k_cluster {
        use crate::n_dimen_algos::distance_between;

        //This function will take the entire list of centroids and also the entire dataset , returns the associated cluster list.
        //i.e which point is present in which cluster respectively.
        pub fn k_cluster(centroids :  Vec<Vec<f32>> , data_set : &[&[f32]]) -> Vec<usize> {

            
            let number_of_features = centroids[0].len();
            let number_of_clusters = centroids.len();
            let number_of_samples = data_set.len();
            //Just some cross checking.
            assert!(centroids[0].len() == data_set[0].len() ,"Both the points(cluster-centroids and sample points) must have the same number of dimensions");

            let mut out_associative_vec = Vec::new();

            //present_nearest:
            //we store the cluster point and it's distance , "if" after the next iteration 
            //we found out that this point is nearer to another centroid we update the present nearest.
            
            //iterating through every point to see for which cluster it belongs to.
            for sample_point in 0..number_of_samples {
                let mut present_nearest : (usize , f32) = (1000, 9000.0);//initialising with obscure values so that this will for sure be updated.
                //now we have one sample in our hand, time to find out the nearest centroid to this.
                for cluster in 0..number_of_clusters {
                    //Now we have a centroid and a sample point ;), time to to find out the distance.
                    let dist_now = distance_between(data_set[sample_point], &centroids[cluster]);
                    
                    if dist_now < present_nearest.1 {
                        present_nearest = (cluster , dist_now);
                    }

                }

                //update the associative array for this point.
                out_associative_vec.push(present_nearest.0);

            }

            out_associative_vec

        } 




    }

}

pub mod n_dimen_algos {
    //takes two arrays of length n(n-dimensional points), returns the euclidean distance between them.
    pub fn distance_between( point_1 : &[f32] , point_2 : &[f32] ) -> f32 {
        
        assert!(point_1.len() >= 1 && point_2.len() >= 1 , "You should have atleast 1 feature for the sample point" );
        assert!(point_1.len() == point_2.len() , "Both the points should have the same number of features");

        let dimensions = point_1.len();
        let mut sum_of_squares_of_difference = 0.0;

        for i in 0..dimensions {
            sum_of_squares_of_difference += (point_1[i] - point_2[i]).powf(2.0);
        }

        sum_of_squares_of_difference.sqrt()

    }
    /*
    this function will take all the points in a certain cluster and returns the average of the points as a point ,
    which should be updated as the new centroid for that cluster.
    */
    pub fn give_avg_point(samples_in_cluster : &[&[f32]]) -> () {

        assert!(samples_in_cluster.len() > 0 , "The cluster must contain atleast one sample point");

        let sum_of_points = 0.0;
        let number_of_features = samples_in_cluster[0].len();
        let number_of_sammples = samples_in_cluster.len();

        
        

        

    } 

    //This is for individual elements this will not be needed for the user to use,
    //takes in one of the feature of all the elements in a cluster and gives out the  
    pub fn average(feature_in_cluster : &[f32]) -> f32 {

        let mut sum = 0_f32;
        let len = feature_in_cluster.len() as f32;
        for i in feature_in_cluster {
            sum += i;
        }

        sum / len

    }

}

pub mod centroid_gen {
    use rand::prelude::*;

    pub fn generate_k_centroids(number_of_clusters : usize ,
                                number_of_features : usize ,
                                lower_limit : f32 , 
                                upper_limit : f32) -> Vec<Vec<f32>> 
    {
        let mut hava = thread_rng();
        //creating an empty array cause now we know the size of the output.
        let mut out_centroids:Vec<Vec<f32>> = Vec::new();
        
        for centroids in 0..number_of_clusters {
            //create this point and push into the all centroids list.
            let mut this_cluster:Vec<f32> = Vec::new();
            for centroid_feature in 0..number_of_features {
                this_cluster.push(hava.gen_range(lower_limit , upper_limit));
            }
            out_centroids.push(this_cluster);
        }

        out_centroids

    }

}





#[cfg(test)]

mod test {
    use crate::{n_dimen_algos::{distance_between, average}, centroid_gen::generate_k_centroids};

    #[test]
    fn distance() {
        
        let p1 = [0.0 , 0.0 , 0.0 , 4.0];
        let p2 = [1.0 , 1.0 , 1.0 , -9.0];

        dbg!(distance_between(&p1, &p2));

    }//working

    #[test]
    fn _average() {
        let p1 = [0.0 , 15.0 , 0.0 , 4.0];

        println!("{}", average(&p1));

    }//working
    
    #[test]
    fn cluster_centroid_generate() {

        println!("{:?}", generate_k_centroids(5, 10, 0.0, 10.0));

    }//working_perfectly!!




}