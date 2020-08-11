#!/bin/bash

declare -a cp_ids=( 1000 2000 3000 4000 5000 )
#6000 7000 8000 9000 10000 )

for cp_id in ${cp_ids[@]}
do
    sed -i "s/cp_id: .*/cp_id: $cp_id/g" ~/exec/anvil/prior/spline/runcards/sample.yml

    echo -n "$cp_id " >> acceptance.txt
    anvil-sample ~/exec/anvil/prior/spline/runcards/sample.yml
done
