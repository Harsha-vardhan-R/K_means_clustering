#![allow(non_snake_case)]

use K_means_clustering::mlalgos::k_cluster::k_means;
fn main() {
    //Example code , of how to use this.
    let _machine = k_means("C:/Users/HARSHA/Downloads/wine-clustering.csv", 12, 0.0, 20.0, 0.01);
    //machine.predict(x);
}
