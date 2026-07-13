package geography.agents;

import org.locationtech.jts.geom.Geometry;

public class Rangeland {
    private final Geometry geometry;

    public Rangeland(Geometry geometry) {
        this.geometry = geometry;
    }

    public Geometry getGeometry() {
        return geometry;
    }
}