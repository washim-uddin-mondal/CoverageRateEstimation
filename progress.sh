printf "%15s %15s %15s %15s\n" "Country" "Fading Shape" "Seed Index" "Iteration"


for country_dir in Results/* 
do 
   country_name=${country_dir:8} 

   if [ $country_name != "Inverse" ]
      then
         for shape_dir in $country_dir/*
            do
              shape_value=$(echo $shape_dir| sed -e 's/.*Shape//')

              fcontent=$(tail $shape_dir/Raw/progress.log)
              seed=$(echo $fcontent | sed -e 's/.*Seed Index: //')
              seed=${seed: 0: 1}
              iteration=$(echo $fcontent | sed -e 's/.*Iteration: //; s/,.*//')
              iteration=${iteration:0:4}

	      printf "%15s %15s %15s %15s\n" $country_name $shape_value $seed $iteration
            done
   fi
done

