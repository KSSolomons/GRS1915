REMOTE_HOST="transfer.ilifu.ac.za"   
REMOTE_BASE="../../scratch3/users/kyles/GRS1915"
LOCAL_DIR="/media/kyle/kyle_phd/GRS1915/FITS/pcalmask"


mkdir -p "$LOCAL_DIR"

for s in {1..12}; do
    echo "========================================="
    echo ">>> Transferring Epoch S${s}..."
    echo "========================================="
    rsync -avzP "kyles@${REMOTE_HOST}:${REMOTE_BASE}/*_S${s}/IMAGES/*pcalmask*-MFS-image.pbcor.fits" "$LOCAL_DIR/"
done

