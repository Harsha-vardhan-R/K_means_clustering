#![allow(non_snake_case, non_camel_case_types, unused_mut, unused_imports)]

#[derive(Debug)]
pub struct sample_point {
    pub data : Vec<f32>,
    pub associated_cluster : Option<u32>,
}

#[derive(Debug)]
pub struct k_means_spec<'a> {
    csv_file_path : &'a str,
    pub header_names : Vec<String>,
    data : Vec<sample_point>,
    pub centroids : Vec<Vec<f32>>,
    k : usize,
    number_of_features : usize,
    number_of_samples : usize,
    threshold : f32,
    pub visuals : bool,
    pub encodings : Option<Vec<String>>,
    varience : Option<Vec<Vec<f32>>>,
    cluster_populations : Option<Vec<usize>>,
}

use core::f32;
use core::num;
use std::dbg;
use std::marker;
use std::string;
use fastrand::Rng;
//use plotters::prelude::*;
use plotlib::page::Page;
use plotlib::repr::{Histogram, HistogramBins, Plot};
use plotlib::style::{BoxStyle, PointMarker, PointStyle};
use plotlib::view::ContinuousView;

impl k_means_spec<'_> {
    
    fn print(&self) {
        println!("{:?}", self );
    }
    ///returns the population of each cluster.
    /// 
    /// '''
    /// let df = k_means("wine-clustering.csv", 3, 10.0, 15.0, 0.01 , vec![]);
    /// let foo = df.print_populations();
    /// '''
    /// 
    /// foo will be of the type vec<usize> and of length df.k.    
    pub fn print_populations(&self) -> Vec<usize> {
        self.cluster_populations.clone().unwrap()
    }    
    //I do not even know if this can be done , we already opened the file once right?
    /* pub fn feature_names(&self) {

    } */
    ///returns the population of each cluster.
    /// 
    /// '''
    /// let df = k_means("wine-clustering.csv", 3, 10.0, 15.0, 0.01 , vec![]);
    /// //The vector should have the same number of elements as the number of clusters.
    ///  
    /// df.encoding_names(vec![String::from("one") , String::from("two") , ....])
    /// '''
    pub fn encoding_names(&mut self , Names : Vec<String>) {
        //checking for the correct number of names 
        assert!(Names.len() == self.k , "The number of names provided do not match the k value");

        self.encodings = Some(Names);

    }
    ///This function is more or less written for debugging , it contains important info but nobody can visualise the data that it outputs.
    /// '''
    /// let df = k_means("wine-clustering.csv", 3, 10.0, 15.0, 0.01 , vec![]);
    /// df.print_associates();
    /// '''
    /// prints each asssociated clusters individually, not pretty to look at.
    pub fn print_associates(&self) {
        for associates in &self.data {
            print!("{}," , associates.associated_cluster.unwrap());
        }
    }
    ///function only works from inside so no need for docs.
    //takes the whole data frame struct and changes the centroid coordinates by finding the AVERAGE of coords in that respective cluster.
    fn update_centroids(&mut self) {
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
            //not a problem if we take random sample points as the initial centroids.
            match avgs_with_counts {
                ( _ , 0) => new_centroids.push(self.centroids[i].clone()),//if the count is zero , we do not modify the centroids.
                _ => new_centroids.push(avgs_with_counts.0.iter().map(|each| each / (avgs_with_counts.1 as f32)).collect()), //else we will do the average thing
            }

        }

        self.centroids = new_centroids
        

    }
    //gives out a vector of variences of each feature in each cluster, and also gives out the number of points in each cluster.
    pub fn get_varience(&mut self) -> Vec<Vec<f32>> {
        //in a cluster the varience = sum((diff(samplepointfeature - associate centroid feature))^2) / number of the samplepoints in that particular cluster.

        //creating the empty and sized vector collection.
        let mut for_varience : Vec<(Vec<f32> , usize)> = vec![(vec![0.0 ; self.number_of_features] , 0_usize) ; self.k];

        for sample_point in &self.data {
        //adding a number to the number of samples in the cluster.
        let this_cluster = sample_point.associated_cluster.unwrap() as usize;
        for_varience[this_cluster].1 += 1;

            for (i , feature) in sample_point.data.iter().enumerate() {
                for_varience[this_cluster].0[i] += (feature - self.centroids[this_cluster][i]).powf(2.0);
            }

        }
        let mut out_vec: Vec<Vec<f32>> = Default::default();

        for ( vector , number_of_samples) in for_varience.iter() {
            if number_of_samples == &0_usize {
                panic!("Cannot calculate without calculating the k means or one of the clusters is empty");
            }

            let mut temp_vec_iter = vec![];

            for feature in vector {
                temp_vec_iter.push(feature / *number_of_samples as f32);
            }
            out_vec.push(temp_vec_iter);
        }

        self.varience = Some(out_vec.clone());

        out_vec

    }
    //to get the variences of different features, normalisation done across different clusters so you will get some zeros for some features
    //minimums are zeros here.
    pub fn get_normal_varience(&mut self) -> Vec<Vec<f32>> {

        //getting the varience vector.
        let varience = if self.varience.is_some() {
            self.varience.clone().unwrap()
        } else {
            self.varience = Some(self.get_varience());
            self.varience.clone().unwrap()
        };

        let mut mod_vec = vec![vec![ 0.0 ; self.number_of_features] ; self.k];

        //Normalising the variences(scaling them down between 0 and 1)
        for feature_index in 0..self.number_of_features {
            let mut temp_vec: Vec<f32> = vec![];

            for i in 0..self.k {   
                temp_vec.push(varience[i][feature_index]);
            }
            temp_vec.sort_by(|a , b| a.partial_cmp(b).unwrap());
            let min = temp_vec[0];
            let max = temp_vec[temp_vec.len() - 1];
            //One feature done.
            //dbg!(max , min);
            let diff = max - min;
            for i in 0..self.k {
                mod_vec[i][feature_index] = (varience[i][feature_index] - min) / diff;
            }

        }

        mod_vec

    }

    pub fn get_weights(&mut self) -> Vec<Vec<f32>> {
        //firstly getting the normalised variences.
        let mut varience_normal = self.get_normal_varience();
        //getting the weights.
        for feature in 0..self.number_of_features {
            let mut sum = 0.0;
            for k in 0..self.k {
                sum += varience_normal[k][feature];
            }
            for k in 0..self.k {
                varience_normal[k][feature] /= sum;
            }
        }
        let mut temp2 = vec![vec![0.0 ; self.number_of_features] ; self.k];
        for (i , cluster) in varience_normal.iter().enumerate() {
            let sum2: f32 = cluster.iter().sum();
            for t in 0..self.number_of_features {
                temp2[i][t] = ((varience_normal[i][t] / sum2) * 10000.0).round() / 100.0;
            }            
        }

        temp2    

    }
    ///'''
    /// the predict function takes a point and gives out the nearest centroid to it.
    /// let df = k_means("wine-clustering.csv", 3, 10.0, 15.0, 0.01 , vec![]);
    /// df.predict();
    /// '''
    /// 
    /// prints the cluster name associated with the nearest centroid.
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

        let pressent_name = match &self.encodings {
            Some(value) => value[closest_centroid_index].to_owned(),
            None => "Encoding names are still not given".to_owned(),
        };
        
        println!("{:?} Belongs to : index -> {} -> Name : {}", x , closest_centroid_index , pressent_name);
        closest_centroid_index as u32
    }
    //here we write the plotting stuff.
    //we need to consider 1 , 2 , 3  or more than 3 features.
    pub fn plot(&self , path : &str) {
        match self.number_of_features {
            1 => (),//This is kinda sketchy, like what are we even doing? we should also plot the centroid positions.
            2 => self.plot_two_dimension(path , 0 , 1),//we will take the names from the indexes, do not worry.
            3 => self.plot_three_dimension(path),
            4..=1000 => self.plot_more_than_three_dimen(path),   
            _  => (),//if your df has more than 1000 features or no features at all, fuck you, Ezekiel!.     
        }    
    }

    //NOTE: This method does not work if you only consider some features  
    
    pub fn get_distributions(&self , path : &str) {
        for i in 0..self.number_of_features {
            let image_name = format!("{}distribution_plot_of_{}.svg" ,path , &self.header_names[i]);
            self.plot_one_dimension(&image_name, i, &self.header_names[i], "intensity/Distribution");
        }
    }
    //this just to get the scatters of the adjacent features ,before clustering
    pub fn get_pre_scatters(&self , path : &str) {
        for i in 0..self.number_of_features - 1 {
            let image_name = format!("{}pre_scatter_plot_between_{}_and_{}.svg",path ,&self.header_names[i], &self.header_names[i + 1]);
            self.plot_two_dimension(&image_name , i , i + 1)
        }
    }

    //private functions for plotting of different number of features.
    pub fn plot_one_dimension(&self , path : &str, feature_index : usize, x_lable : &str, y_label : &str) {
        //first we will get the dataset then we can set the graph scale
        let mut data_to_plot: Vec<(f64 , f64)> = vec![ (0.0 , 0.0) ; self.number_of_samples];
        let mut min = 100000 as f64;
        let mut max = -100 as f64;//temporary////be careful values obviously can go lower than that.
        for (index, sample_point) in self.data.iter().enumerate() {
            data_to_plot[index] = (sample_point.data[feature_index].clone().try_into().unwrap(), 0.0);
            if data_to_plot[index].0 > max {
                max = data_to_plot[index].0;
            } else if data_to_plot[index].0 < min {
                min = data_to_plot[index].0;
            }
        }

        /* let number_of_divisions = 30;//This is my code for putting the data in histogram bins but the plotlib library already contains this so, we will directly use it.
        let mut distribution_vector = vec![0.0 as f64; number_of_divisions];
        let gradient = (max - min) / number_of_divisions as f64;
        let mut set = 0;
        for i in &data_to_plot {
            set = 0;
            for n in 0..number_of_divisions { 
                set += 1;             
                if i.0 < ((gradient * set as f64) + min) {
                    print!("{}  ,  ", &i.0);
                    print!("{}\n", (gradient * n as f64) + min);
                    break;
                }
            }
            distribution_vector[set - 1] += 1.0;
        }

        dbg!(&distribution_vector);
        let sum: f64 = distribution_vector.iter().sum();
        dbg!(sum); */
        let mut temppp = vec![];
        for i in data_to_plot.iter() {
            temppp.push(i.0);
        }

        let h  = Histogram::from_slice(temppp.as_slice(), HistogramBins::Count(15))
            .style(&BoxStyle::new().fill("burlywood"));//the count needs to be taken care of,I think we should change it according to the data.
        let mut max_on_y = f64::MIN;
        for i in h.bin_counts.iter() {
            if *i > max_on_y {
                max_on_y = *i;
            }
        }

        max_on_y = (max_on_y * 1.1).floor();//this is to give a little amount of the head space in the plot.
 
        /* let s1: Plot = Plot::new(data_to_plot).point_style(
            PointStyle::new()
            .marker(PointMarker::Square)
            .colour("FFFFFF"),
        ); *///this looks like shit .

        let v = ContinuousView::new()
            .add(h)
            .x_range(min, max)
            .y_range(0.0, max_on_y)
            .x_label(x_lable)
            .y_label(y_label);

        Page::single(&v).save(path).unwrap();
        
    }

    pub fn plot_two_dimension(&self , path : &str , feature_index_1 : usize , feature_index_2 : usize ) {
        let mut data_to_plot: Vec<(f64 , f64)> = vec![ (0.0 , 0.0) ; self.number_of_samples];
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::MIN;//temporary////be careful values obviously can go lower than that.
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::MIN;
        for (index, sample_point) in self.data.iter().enumerate() {
            data_to_plot[index] = (sample_point.data[feature_index_1].clone().try_into().unwrap(), sample_point.data[feature_index_2].clone().try_into().unwrap());
            if data_to_plot[index].0 > max_x {
                max_x = data_to_plot[index].0;
            } else if data_to_plot[index].0 < min_x {
                min_x = data_to_plot[index].0;
            }
            if data_to_plot[index].1 > max_y {
                max_y = data_to_plot[index].1;
            } else if data_to_plot[index].1 < min_y {
                min_y = data_to_plot[index].1;
            }
        }

        dbg!(min_x,max_x,min_y,max_y);

        // We create our scatter plot from the data
        let s1: Plot = Plot::new(data_to_plot).point_style(
            PointStyle::new()
                .marker(PointMarker::Circle) // setting the marker to be a square
                .colour("#DD3355"),
        ); 

        // The 'view' describes what set of data is drawn
        let v = ContinuousView::new()
            .add(s1)
            .x_range( min_x , max_x )
            .y_range( min_y , max_y )
            .x_label(self.header_names[feature_index_1].clone())
            .y_label(self.header_names[feature_index_2].clone());

        // A page with a single view is then saved to an SVG file
        Page::single(&v).save(path).unwrap();

        }

    fn plot_three_dimension(&self , path : &str) {

    }

    fn plot_more_than_three_dimen(&self , path : &str) {

    }
    

}
//This is the main logic behind, user will use this.
//Lower_limit and upper limit will be used in the random generation function.
///'''
///                                                           / This is a feature you, can select which features you want to consider while training, if it is empty all the features will be considered.
///address of the csv data file-^                            ^      
///let mut machine = k_means("IRIS.csv", 3, None, 0.001 , vec![0,1,2,3]);
///                                      ^    ^     ^this is the threshold value , that is the limiting value of maximum movement by any centroid while breaking out.
///                                      |    \----this part can be a none or a some((f32, f32))
///                                      \-----the number of clusters you want to form.
/// if it is some then , we will generate a set of centroids for this data frame, whose fields are randomly generated.
/// '''
pub fn k_means( csv_file_path: &str,
                k_value: usize,
                limits: Option<(f32, f32)>,
                Threshold: f32,
                which_features: Vec<usize>) -> k_means_spec {

    //mut, because we will change the centroid values, after every iteration.
    let mut present_dataFrame = new_df(csv_file_path, k_value, Threshold, limits, which_features);//now we have a data_set and its specifications to work on.
    let mut count = 1;
    //clustering in k means until we get the centroid points moving less than threshold value after one iteration.
    //main loop
    loop {
        //saving the points , to calculate the distance afterwards.
        let previous_centroids = present_dataFrame.centroids.clone();
        //debugging
        //dbg!(&present_dataFrame.centroids);
        
        k_cluster(&mut present_dataFrame);
        //changing the centroids based on the present sample points association, 
        //this is a method call directly implemented on the K_means_spec.
        present_dataFrame.update_centroids();

        //if the largest change between any centroid respective to its previous position is leaa than the threshold value, we will break out of the loop.
        let max_moved = max_distance_between_sets(&previous_centroids , &present_dataFrame.centroids);
        if  max_moved < present_dataFrame.threshold {
            println!("Max change while breaking out = {max_moved}");
            println!("Done!");
            break;
        } else {
            println!("Max change in position of any centroid = {max_moved}");
        }

        //debugging
        //present_dataFrame.print_associates();                
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

    let mut to_be_filled = vec![ 0 ; total_df.k ];
     
    for sample_point in 0..total_df.number_of_samples {
        let mut present_nearest : (usize , f32) = (1000 , std::f32::INFINITY);//initialising with obscure values so that this will for sure be updated.
        //now we have one sample in our hand, time to find out the nearest centroid to this.
        for cluster in 0..total_df.k {
            //Now we have a centroid and a sample point ;), time to to find out the distance.
            let dist_now = distance_between(&total_df.data[sample_point].data, &total_df.centroids[cluster]);
            
            if dist_now < present_nearest.1 {
                present_nearest = (cluster , dist_now);
            }

        }
        //storing the nearest centroid in the associated cluster field.
        total_df.data[sample_point].associated_cluster = Some(present_nearest.0 as u32);
        //updating the cluster_population field.
        to_be_filled[present_nearest.0] += 1;

    }

    total_df.cluster_populations = Some(to_be_filled);
    

} 


//This only generates random numbers , but do not worry about different ranges in different features, even if it contains one sample point,
//while calculating the average they will set , keep your fingers crossed for it atleast catches one sample point.maybe will update in next version.
fn generate_k_centroids(number_of_clusters : usize ,
                        number_of_features : usize ,
                        lower_limit : f32 , 
                        upper_limit : f32) -> Vec<Vec<f32>> {
    
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

//Private function
//randomly selects some points in the data sets ,to be taken as the initial centroid positions.
//and obviously it cannot produce two same points.
fn get_random_samples_from_df(data : &Vec<sample_point>, k: usize , number_of_features : usize, number_of_samples : usize) -> Vec<Vec<f32>> {
    assert!(number_of_samples >= k, "You cannot have k greater than the number of all the points");
    //here we randomly generate indexes without repeating.
    let mut rand_index = vec![];
    while rand_index.len() < k {
        let mut rng = Rng::new();
        let random_u32 = rng.u32(0..number_of_samples as u32);
        if rand_index.contains(&random_u32) {
            continue;
        } else {
            rand_index.push(random_u32);
        }
    }
    let mut centroid: Vec<Vec<f32>> = vec![ vec![0.0 ; number_of_features] ; k];
    dbg!(&rand_index);
    //dbg!(&centroid);
    for (index , i) in rand_index.into_iter().enumerate() {
        for j in 0..number_of_features {
            centroid[index][j] = data[i as usize].data[j];
        }
    }
    dbg!(&centroid);
    centroid
}

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use csv::ReaderBuilder;

use crate::n_dimen::distance_between;
use crate::n_dimen::max_distance_between_sets;

fn csv_to_df(
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
                //This is really safe, 'cause if the object cannot be parsed , then it will be not considered an error because of the .ok() , which returns None if there is an error.
                // Consider only the columns specified by which_features/also problematic, if the data has some null or NaN's , we are fucked!
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
///warning this does not implement only selected features implementation, please! this is not even experimental.
fn get_headers(path : &str) -> Vec<String> {
    let file_system = File::open(path).unwrap();
    let mut out_vector : Vec<String> = vec![];
    let reader = BufReader::new(file_system);
    let mut csv_header = ReaderBuilder::new().has_headers(false).from_reader(reader);
    for header in csv_header.records() {
        let head = header.unwrap();
        for i in head.iter() {
            out_vector.push(i.to_owned());
        }
        break;
    }

    out_vector
}

//creating a struct, which stores all the info about the present k_mean.
//private function.
fn new_df(csv_file_path : & str ,K : usize, threshold : f32 ,limits: Option<(f32, f32)> , which_features: Vec<usize>) -> k_means_spec { 
    let data = csv_to_df(csv_file_path , which_features).unwrap();
    //we are calculating the number of features after making the data frame, so we need not change the size while generating the centroids.
    let number_of_features = data[0].data.len();
    let number_of_samples = data.len();
    //creating and returning a new k_means_spec struct.
    k_means_spec {  csv_file_path: csv_file_path,
                    centroids : match limits {
                        //we will match the limits, if the user gives none, then we will take some random points as initial centroids.
                        Some((lower_limit , upper_limit)) => generate_k_centroids(K, number_of_features, lower_limit, upper_limit),
                        None => get_random_samples_from_df(&data, K, number_of_features, number_of_samples),
                    },
                    header_names : get_headers(csv_file_path),
                    data: data,                       
                    k: K,
                    number_of_features: number_of_features,
                    number_of_samples: number_of_samples,
                    threshold : threshold,
                    visuals: true,
                    encodings : None,
                    varience: None,
                    cluster_populations : None,
    }
}