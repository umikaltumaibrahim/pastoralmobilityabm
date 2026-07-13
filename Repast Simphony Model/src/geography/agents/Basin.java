package geography.agents;

import org.locationtech.jts.geom.Geometry;

public class Basin {
    private final Geometry geometry;

    public Basin(Geometry geometry) {
        this.geometry = geometry;
    }

    public Geometry getGeometry() {
        return geometry;
    }
}