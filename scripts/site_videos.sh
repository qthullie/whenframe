#!/bin/sh
# The three films as small videos for the demo site, 640 pixels wide: OUT/NAME.mp4 (H.264, for every
# browser) and OUT/NAME.webm (VP9, for browsers built without H.264). The site plays them from the
# moment a search found.
#
#     scripts/site_videos.sh [OUT [WORK]]      # OUT: site/videos (default); WORK: the films, downloaded if missing
#
# The films come from download.blender.org: Big Buck Bunny, Sintel and Tears of Steel, (c) Blender
# Foundation, CC BY 3.0, the same files the site's frames were indexed from (scripts/site.py), so the
# times match. The deploy workflow (.github/workflows/pages.yml) runs this; the videos stay out of git.
set -eu
out=${1:-site/videos}
work=${2:-$(mktemp -d)}
mkdir -p "$out" "$work"

fetch() {  # film, its path on download.blender.org; a .zip is unpacked
    [ -s "$work/$1" ] && return
    curl -sSfL --retry 5 -o "$work/download" "https://download.blender.org/$2"
    case $2 in
        *.zip) unzip -o -q -j "$work/download" -d "$work" && rm "$work/download" ;;
        *) mv "$work/download" "$work/$1" ;;
    esac
}
fetch BigBuckBunny_640x360.m4v peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v.zip
fetch Sintel.2010.720p.mkv durian/movies/Sintel.2010.720p.mkv.zip
fetch tears_of_steel_720p.mov demo/movies/ToS/tears_of_steel_720p.mov

encode() {  # source file, name
    [ -s "$out/$2.mp4" ] || ffmpeg -v error -nostdin -y -i "$work/$1" -vf scale=640:-2 -g 120 \
        -c:v libx264 -preset veryfast -crf 30 -profile:v main -pix_fmt yuv420p \
        -c:a aac -b:a 64k -ac 2 -movflags +faststart "$out/$2.mp4"
    [ -s "$out/$2.webm" ] || ffmpeg -v error -nostdin -y -i "$work/$1" -vf scale=640:-2 -g 120 \
        -c:v libvpx-vp9 -b:v 0 -crf 40 -deadline realtime -cpu-used 8 -row-mt 1 \
        -c:a libopus -b:a 48k -ac 2 "$out/$2.webm"
    echo "$2: $(du -h "$out/$2.mp4" | cut -f1) mp4, $(du -h "$out/$2.webm" | cut -f1) webm"
}
encode BigBuckBunny_640x360.m4v big-buck-bunny
encode Sintel.2010.720p.mkv sintel
encode tears_of_steel_720p.mov tears-of-steel
