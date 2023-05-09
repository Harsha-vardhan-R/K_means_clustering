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