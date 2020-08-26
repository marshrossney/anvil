#!/bin/bash

declare -a cp_ids=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 )
#declare -a cp_ids=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 )

for cp_id in ${cp_ids[@]}
do
    sed -i "s/cp_id: .*/cp_id: $cp_id/g" runcards/sample.yml

    echo -n "$cp_id " >> acceptance.txt
    anvil-sample runcards/sample.yml
done

sed -i "s/cp_id: .*/cp_id: -1/g" runcards/sample.yml
