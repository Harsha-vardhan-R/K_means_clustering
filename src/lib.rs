#![allow(non_snake_case, non_camel_case_types, unused_mut, unused_imports , dead_code )]
/*
Mostly using vectors because, using arrays really makes everything complicated and less flexible, 
we are just using the indexing feature and not continuouslly changing the size of the vector,
so we do not have any kind of considerable performance difference.
*/

#[cfg(test)]
mod test {
    

    #[test]
    fn distance() {
        
        let p1 = vec![0.0 , 0.0 , 0.0 , 4.0];
        let p2 = vec![1.0 , 1.0 , 1.0 , -9.0];

        //dbg!(distance_between(&p1, &p2));

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