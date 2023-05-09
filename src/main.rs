#![allow(non_snake_case)]
pub mod K_Means_Clustering {
    pub mod k_means_clustering;
}
pub mod n_dimen;

use std::time;

use crate::K_Means_Clustering::k_means_clustering::k_means;



fn main() {
    //Example code.
    //creating the data frame and respective , specifications.
    let start_time = time::Instant::now();
    let machine = k_means("wine-clustering.csv", 3, 10.0, 15.0, 0.01 , vec![]);
    //here we provide the names for previously stored just as k - encodings.//still not done.
    //machine.encoding_names();
    //predict.    still accuracy and testing functions are not written.
    machine.predict(&vec![ 14.23 , 1.71 , 2.43 , 15.6 , 127_f32 , 2.8 , 3.06 , 0.28 , 2.29 , 5.64 , 1.04 , 3.92  , 1065.0 ]);
    print!("Time Taken: ");
    print!("{:?}\n", time::Instant::now() - start_time);
}
