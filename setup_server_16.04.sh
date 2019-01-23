#!/usr/bin/env bash

apt_update_called=false
# TODO break into a function to check and add to list if needed
# then call update if that list is non-empty, and then install all at once (each
# invokation of apt install seems kinda slow, plus more verbose)
function check_and_install {
    package_name="$1"
    if ! dpkg-query -W -f='${Status}' ${package_name} | grep "ok installed";
    then
        if $apt_update_called; then
            sudo apt update
            apt_update_called=true
        fi
        sudo apt install -y ${package_name}
    fi
}

declare -a required_packages=("postgresql"
                              "libpq-dev"
                              "postgresql-client"
                              "postgresql-client-common")

for p in "${required_packages[@]}"
do
    check_and_install "$p"
done

# TODO how to best automatically add user? auth?
sudo adduser tracedb --gecos "" --disabled-password
echo "tracedb:tracedb" | sudo chpasswd

sudo -u postgres psql -f create_db_and_role.sql
sudo -u tracedb psql -f setup.sql
