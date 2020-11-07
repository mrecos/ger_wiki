library(tidyverse)

nom_places <- read_csv("../data/results/nom_places.csv")
nom_counts <- nom_places %>%
    group_by(place) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    filter(count > 100) %>%
    {
        rbind(head(.), "...", tail(.))
    }
print(nom_counts)


nam_places <- read_csv("../data/results/nam_places.csv")
nam_counts <- nam_places %>%
    group_by(place) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    filter(count > 100) %>%
    {
        rbind(head(.), "...", tail(.))
    }
print(nam_counts)
