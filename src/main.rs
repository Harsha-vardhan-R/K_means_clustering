#![allow(non_snake_case)]

use K_means_clustering::mlalgos::k_cluster::k_means;
fn main() {
    //Example code.
    //creating the data frame and respective , specifications.
    let machine = k_means("C:/Users/HARSHA/Downloads/wine-clustering.csv", 3, 10.0, 15.0, 0.01);
    //here we provide the names for previously stored just as k - encodings.
    //machine.encoding_names();
    //predict.    still accuracy and testing functions are not written.
    //machine.predict();
}
