# *** IMPORTANT ***
# Super use 'sudo' permissions will be needed to run this script
# e.g. sudo setup_preprocess.sh

##############################


# copy preprocess script to bin
cp preprocess.sh /usr/bin/.

#write out current crontab
crontab -l > cron_jobs
#echo preprocess cron job into cron file
echo "0 22 * * * preprocess.sh" >> cron_jobs
#install new cron file
crontab cron_jobs
rm cron_jobs
