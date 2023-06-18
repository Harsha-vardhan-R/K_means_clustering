#[cfg(test)]

mod k_means_clustering_tests {

use std::time;

use crate::K_Means_Clustering::k_means_clustering::k_means;



#[test]

fn k_means_test_wine() {
    let start_time = time::Instant::now();

    //let mut machine = k_means("wine-clustering.csv", 3, Some((10.0, 15.0)), 0.01 , vec![]);
    let mut machine = k_means("src/K_Means_Clustering/wine-clustering.csv", 3, None, 0.01 , vec![]);
    machine.get_distributions("src/");
    //here we provide the names for previously stored just as k - encodings.//still not done.
    //machine.encoding_names();
    //predict.    still accuracy and testing functions are not written.
    machine.encoding_names(vec!["cluster_1".to_owned() , "cluster_2".to_owned() , "cluster_3".to_owned()]);
    machine.predict(&vec![ 14.23 , 1.71 , 2.43 , 15.6 , 127_f32 , 2.8 , 3.06 , 0.28 , 2.29 , 5.64 , 1.04 , 3.92  , 1065.0 ]);

    dbg!(&machine.get_varience());    
    dbg!(&machine.get_weights());
    dbg!(&machine.centroids);
    dbg!(&machine.print_populations());
    print!("Time Taken: ");
    print!("{:?}\n", time::Instant::now() - start_time);
}//getting [69,62,47], 

#[test]

fn k_means_test_iris() {
    let start_time = time::Instant::now();

    //let mut machine = k_means("IRIS.csv", 3, Some((0.0, 8.0)), 0.001 , vec![0,1,2,3]);
    let mut machine = k_means("src/K_Means_Clustering/IRIS.csv", 3, None, 0.001 , vec![0,1,2,3]);
    //dbg!(&machine.header_names);
    //machine.get_distributions();
    //machine.get_pre_scatters("src/");
    machine.get_post_scatters("src/");
    dbg!(&machine.print_populations());
    dbg!(&machine.get_varience());
    dbg!(&machine.get_weights());
    dbg!(&machine.centroids);
    machine.plot_one_dimension("src/K_Means_Clustering/hahaha.png" , 0, "   ", "   ");

    print!("Time Taken: ");
    print!("{:?}\n", time::Instant::now() - start_time);
    
}//getting [50,39,61].

}

