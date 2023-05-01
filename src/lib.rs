#![allow(non_snake_case)]

pub mod mlalgos {
    
    pub mod k_cluster {
        //This function will take the entire list of centroids and also the entire dataset , returns the associated cluster list.
        pub fn k_cluster(centroids : [[f32]] , data_set : [[f32]] ) -> [usize] {

        }

    }

}

pub mod n_dimen_algos {
    //takes two arrays of length n, returns the euclidean distance between them.
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
    pub fn give_avg_point() -> [f32] {

    }

    //This is for individual elements this will not be needed for the user to use
    pub fn average() {

    }

}



#[cfg(test)]

mod test {
    use crate::n_dimen_algos::distance_between;


    #[test]
    fn distance() {
        
        let p1 = [0.0 , 0.0 , 0.0 , 4.0];
        let p2 = [1.0 , 1.0 , 1.0 , -9.0];

        dbg!(distance_between(&p1, &p2));

    }
}